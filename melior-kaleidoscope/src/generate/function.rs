use super::{error, module::ModuleGenerator};
use crate::{
    ast::{
        self, Binary, BinaryOperator, Call, Expression, Literal, Return, Statement, StructLiteral,
        VariableDeclaration, VariableDefinition,
    },
    error::Error,
    toy,
};
use melior::ir::{
    Attribute, Block, BlockLike, Type, Value, ValueLike,
    attribute::{ArrayAttribute, DenseElementsAttribute, FlatSymbolRefAttribute, IntegerAttribute},
    r#type::{IntegerType, RankedTensorType, TupleType},
};
use std::collections::HashMap;

pub struct FunctionGenerator<'c, 'a, 'b> {
    module: &'b ModuleGenerator<'c, 'a>,
    block: &'b Block<'c>,
    scopes: Vec<HashMap<&'a str, (Value<'c, 'b>, &'a VariableDeclaration)>>,
}

impl<'c, 'a, 'b> FunctionGenerator<'c, 'a, 'b> {
    pub fn new(module: &'b ModuleGenerator<'c, 'a>, block: &'b Block<'c>) -> Self {
        Self {
            module,
            block,
            scopes: vec![Default::default()],
        }
    }

    pub fn declare(
        &mut self,
        variable: &'a VariableDeclaration,
        value: Value<'c, 'b>,
    ) -> Result<(), Error> {
        if self.lookup(variable.name()).is_some() {
            return Err(error(
                variable.location(),
                format!("variable '{}' is already declared", variable.name()),
            ));
        }

        self.scopes
            .last_mut()
            .expect("at least one scope")
            .insert(variable.name(), (value, variable));

        Ok(())
    }

    fn lookup(&self, name: &str) -> Option<(Value<'c, 'b>, &'a VariableDeclaration)> {
        self.scopes
            .iter()
            .rev()
            .find_map(|scope| scope.get(name).copied())
    }

    pub fn generate_block(&mut self, statements: &'a [Statement]) -> Result<(), Error> {
        self.scopes.push(Default::default());

        for statement in statements {
            match statement {
                Statement::Expression(expression) => {
                    self.generate_expression(expression)?;
                }
                Statement::Print(print) => {
                    self.block.append_operation(
                        toy::print(
                            self.module.context(),
                            self.generate_expression(print.argument())?,
                            self.module.location(print.location()),
                        )
                        .into(),
                    );
                }
                Statement::Return(r#return) => {
                    self.generate_return(r#return)?;
                    break;
                }
                Statement::VariableDefinition(definition) => {
                    self.generate_variable_definition(definition)?;
                }
            }
        }

        self.scopes.pop();

        Ok(())
    }

    fn generate_return(&mut self, r#return: &'a Return) -> Result<(), Error> {
        self.block.append_operation(
            toy::r#return(
                self.module.context(),
                r#return
                    .value()
                    .map(|value| self.generate_expression(value))
                    .transpose()?
                    .as_slice(),
                self.module.location(r#return.location()),
            )
            .into(),
        );

        Ok(())
    }

    fn generate_variable_definition(
        &mut self,
        definition: &'a VariableDefinition,
    ) -> Result<(), Error> {
        let declaration = definition.declaration();
        let value = self.generate_expression(definition.initializer())?;
        let value = match declaration.r#type() {
            ast::Type::Struct(_) => {
                let r#type = self.module.variable_type(declaration)?;

                if r#type != value.r#type() {
                    return Err(error(
                        declaration.location(),
                        format!(
                            "struct type of initializer is different than the variable declaration: expected {type} but got {}",
                            value.r#type()
                        ),
                    ));
                }

                value
            }
            // A reshape operation is emitted for a declaration with a shape. It
            // gets optimized out later as needed.
            ast::Type::Tensor(shape) if !shape.is_empty() => self
                .block
                .append_operation(
                    toy::reshape(
                        self.module.context(),
                        self.module.tensor_type(shape, declaration.location())?,
                        value,
                        self.module.location(declaration.location()),
                    )
                    .into(),
                )
                .result(0)?
                .into(),
            ast::Type::Tensor(_) => value,
        };

        self.declare(declaration, value)
    }

    fn generate_expression(&mut self, expression: &'a Expression) -> Result<Value<'c, 'b>, Error> {
        match expression {
            Expression::Binary(binary) => self.generate_binary(binary),
            Expression::Call(call) => self.generate_call(call),
            Expression::Literal(literal) => Ok(self
                .block
                .append_operation(
                    toy::constant(
                        self.module.context(),
                        self.module
                            .tensor_type(literal.dimensions(), literal.location())?,
                        self.literal_attribute(literal)?,
                        self.module.location(literal.location()),
                    )
                    .into(),
                )
                .result(0)?
                .into()),
            Expression::Number(number) => {
                let r#type = RankedTensorType::new(&[], Type::float64(self.module.context()), None);

                Ok(self
                    .block
                    .append_operation(
                        toy::constant(
                            self.module.context(),
                            r#type.into(),
                            DenseElementsAttribute::f64_values(r#type.into(), &[number.value()])?,
                            self.module.location(number.location()),
                        )
                        .into(),
                    )
                    .result(0)?
                    .into())
            }
            Expression::StructLiteral(literal) => {
                let (attribute, r#type) = self.struct_literal_attribute(literal)?;

                Ok(self
                    .block
                    .append_operation(
                        toy::struct_constant(
                            self.module.context(),
                            r#type,
                            attribute,
                            self.module.location(literal.location()),
                        )
                        .into(),
                    )
                    .result(0)?
                    .into())
            }
            Expression::Variable(variable) => self
                .lookup(variable.name())
                .map(|(value, _)| value)
                .ok_or_else(|| {
                    error(
                        variable.location(),
                        format!("unknown variable '{}'", variable.name()),
                    )
                }),
        }
    }

    fn generate_binary(&mut self, binary: &'a Binary) -> Result<Value<'c, 'b>, Error> {
        let lhs = self.generate_expression(binary.lhs())?;
        let location = self.module.location(binary.location());

        if binary.operator() == BinaryOperator::Access {
            let Some(index) = self.member_index(binary) else {
                return Err(error(
                    binary.location(),
                    "invalid access into struct expression".into(),
                ));
            };

            return Ok(self
                .block
                .append_operation(
                    toy::struct_access(
                        self.module.context(),
                        TupleType::try_from(lhs.r#type())?.r#type(index)?,
                        lhs,
                        IntegerAttribute::new(
                            IntegerType::new(self.module.context(), 64).into(),
                            index as i64,
                        ),
                        location,
                    )
                    .into(),
                )
                .result(0)?
                .into());
        }

        let rhs = self.generate_expression(binary.rhs())?;
        let output = self.module.unranked_tensor_type();

        Ok(self
            .block
            .append_operation(match binary.operator() {
                BinaryOperator::Add => {
                    toy::add(self.module.context(), output, lhs, rhs, location).into()
                }
                BinaryOperator::Multiply => {
                    toy::mul(self.module.context(), output, lhs, rhs, location).into()
                }
                BinaryOperator::Access | BinaryOperator::Subtract => {
                    return Err(error(
                        binary.location(),
                        format!("invalid binary operator '{}'", binary.operator()),
                    ));
                }
            })
            .result(0)?
            .into())
    }

    fn struct_definition(&self, expression: &Expression) -> Option<&'a ast::Struct> {
        self.module.struct_definition(match expression {
            Expression::Variable(variable) => match self.lookup(variable.name())?.1.r#type() {
                ast::Type::Struct(name) => name,
                ast::Type::Tensor(_) => return None,
            },
            Expression::Binary(binary) if binary.operator() == BinaryOperator::Access => {
                let Expression::Variable(member) = binary.rhs() else {
                    return None;
                };

                match self
                    .struct_definition(binary.lhs())?
                    .variables()
                    .iter()
                    .find(|variable| variable.name() == member.name())?
                    .r#type()
                {
                    ast::Type::Struct(name) => name,
                    ast::Type::Tensor(_) => return None,
                }
            }
            _ => return None,
        })
    }

    fn member_index(&self, access: &Binary) -> Option<usize> {
        let Expression::Variable(member) = access.rhs() else {
            return None;
        };

        self.struct_definition(access.lhs())?
            .variables()
            .iter()
            .position(|variable| variable.name() == member.name())
    }

    fn generate_call(&mut self, call: &'a Call) -> Result<Value<'c, 'b>, Error> {
        let location = self.module.location(call.location());
        let arguments = call
            .arguments()
            .iter()
            .map(|argument| self.generate_expression(argument))
            .collect::<Result<Vec<_>, _>>()?;

        if call.callee() == "transpose" {
            let [argument] = arguments.as_slice() else {
                return Err(error(
                    call.location(),
                    "function 'transpose' expects exactly one argument".into(),
                ));
            };

            return Ok(self
                .block
                .append_operation(
                    toy::transpose(
                        self.module.context(),
                        self.module.unranked_tensor_type(),
                        *argument,
                        location,
                    )
                    .into(),
                )
                .result(0)?
                .into());
        }

        let Some(function_type) = self.module.function_type(call.callee()) else {
            return Err(error(
                call.location(),
                format!("unknown function '{}'", call.callee()),
            ));
        };

        if function_type.result_count() != 1 {
            return Err(error(
                call.location(),
                format!("function '{}' does not return a value", call.callee()),
            ));
        }

        Ok(self
            .block
            .append_operation(
                toy::generic_call(
                    self.module.context(),
                    function_type.result(0)?,
                    &arguments,
                    FlatSymbolRefAttribute::new(self.module.context(), call.callee()),
                    location,
                )
                .into(),
            )
            .result(0)?
            .into())
    }

    fn literal_attribute(&self, literal: &Literal) -> Result<DenseElementsAttribute<'c>, Error> {
        let mut data = vec![];

        collect_literal_data(literal, &mut data);

        Ok(DenseElementsAttribute::f64_values(
            self.module
                .ranked_tensor_type(literal.dimensions(), literal.location())?
                .into(),
            &data,
        )?)
    }

    fn struct_literal_attribute(
        &self,
        literal: &StructLiteral,
    ) -> Result<(ArrayAttribute<'c>, Type<'c>), Error> {
        let (attributes, types) = literal
            .values()
            .iter()
            .map(|value| {
                Ok(match value {
                    Expression::Literal(literal) => (
                        Attribute::from(self.literal_attribute(literal)?),
                        self.module.unranked_tensor_type(),
                    ),
                    Expression::Number(number) => (
                        DenseElementsAttribute::f64_values(
                            RankedTensorType::new(&[], Type::float64(self.module.context()), None)
                                .into(),
                            &[number.value()],
                        )?
                        .into(),
                        self.module.unranked_tensor_type(),
                    ),
                    Expression::StructLiteral(literal) => {
                        let (attribute, r#type) = self.struct_literal_attribute(literal)?;

                        (attribute.into(), r#type)
                    }
                    _ => {
                        return Err(error(
                            value.location(),
                            "literal expected in struct literal".into(),
                        ));
                    }
                })
            })
            .collect::<Result<(Vec<_>, Vec<_>), Error>>()?;

        Ok((
            ArrayAttribute::new(self.module.context(), &attributes),
            TupleType::new(self.module.context(), &types).into(),
        ))
    }
}

fn collect_literal_data(literal: &Literal, data: &mut Vec<f64>) {
    for value in literal.values() {
        match value {
            Expression::Literal(literal) => collect_literal_data(literal, data),
            Expression::Number(number) => data.push(number.value()),
            _ => {}
        }
    }
}
