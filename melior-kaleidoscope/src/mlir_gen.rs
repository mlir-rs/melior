//! MLIR generation from the Toy AST into the Toy dialect.

use crate::{
    ast::{
        self, Binary, BinaryOperator, Call, Expression, Function, Literal, Record, Return,
        Statement, StructLiteral, VariableDeclaration, VariableDefinition,
    },
    error::Error,
    toy,
    utility::{is_operation, terminator},
};
use melior::{
    Context,
    ir::{
        Block, BlockLike, Location, Module, Region, RegionLike, Type, Value, ValueLike,
        attribute::{
            ArrayAttribute, DenseElementsAttribute, FlatSymbolRefAttribute, IntegerAttribute,
            StringAttribute, TypeAttribute,
        },
        operation::OperationLike,
        r#type::{FunctionType, IntegerType, RankedTensorType, TupleType},
    },
};
use std::collections::HashMap;

/// Generates an MLIR module in the Toy dialect from a Toy AST.
pub fn generate<'c>(context: &'c Context, module: &ast::Module) -> Result<Module<'c>, Error> {
    ModuleGenerator::new(context, module.path()).generate(module)
}

struct ModuleGenerator<'c, 'a> {
    context: &'c Context,
    path: &'a str,
    functions: HashMap<&'a str, FunctionType<'c>>,
    structs: HashMap<&'a str, (TupleType<'c>, &'a ast::Struct)>,
}

impl<'c, 'a> ModuleGenerator<'c, 'a> {
    fn new(context: &'c Context, path: &'a str) -> Self {
        Self {
            context,
            path,
            functions: Default::default(),
            structs: Default::default(),
        }
    }

    fn generate(mut self, module: &'a ast::Module) -> Result<Module<'c>, Error> {
        let mlir_module = Module::new(Location::unknown(self.context));

        for record in module.records() {
            match record {
                Record::Function(function) => {
                    let operation = self.generate_function(function)?;

                    self.functions.insert(
                        function.prototype().name(),
                        FunctionType::try_from(operation.function_type()?.value())?,
                    );
                    mlir_module.body().append_operation(operation.into());
                }
                Record::Struct(r#struct) => self.generate_struct(r#struct)?,
            }
        }

        if !mlir_module.as_operation().verify() {
            return Err(Error::Verification);
        }

        Ok(mlir_module)
    }

    fn generate_struct(&mut self, r#struct: &'a ast::Struct) -> Result<(), Error> {
        if self.structs.contains_key(r#struct.name()) {
            return Err(error(
                r#struct.location(),
                format!("struct type with name '{}' already exists", r#struct.name()),
            ));
        }

        let types = r#struct
            .variables()
            .iter()
            .map(|variable| {
                if matches!(variable.r#type(), ast::Type::Tensor(shape) if !shape.is_empty()) {
                    return Err(error(
                        variable.location(),
                        "variables within a struct definition must not have shapes".into(),
                    ));
                }

                self.variable_type(variable)
            })
            .collect::<Result<Vec<_>, _>>()?;

        self.structs.insert(
            r#struct.name(),
            (TupleType::new(self.context, &types), r#struct),
        );

        Ok(())
    }

    fn generate_function(
        &mut self,
        function: &'a Function,
    ) -> Result<toy::FuncOperation<'c>, Error> {
        let prototype = function.prototype();
        let location = self.location(prototype.location());
        let argument_types = prototype
            .parameters()
            .iter()
            .map(|parameter| self.variable_type(parameter))
            .collect::<Result<Vec<_>, _>>()?;
        let block = Block::new(
            &argument_types,
            &prototype
                .parameters()
                .iter()
                .map(|parameter| self.location(parameter.location()))
                .collect::<Vec<_>>(),
        )?;

        let mut generator = FunctionGenerator::new(self, &block);

        for (index, parameter) in prototype.parameters().iter().enumerate() {
            generator.declare(parameter, block.argument(index)?.into())?;
        }

        generator.generate_block(function.body())?;

        // A result type of a function is inferred from its return statement.
        let result_types = if let Some(terminator) = terminator(&block)
            && is_operation(terminator, "toy.return")
        {
            terminator.operands().map(|value| value.r#type()).collect()
        } else {
            block.append_operation(toy::r#return(self.context, &[], location).into());
            vec![]
        };

        let region = Region::new();
        region.append_block(block);

        // Functions other than the main one are private so that they are
        // removed after inlining.
        Ok(toy::FuncOperation::builder(self.context, location)
            .body(region)
            .sym_name(StringAttribute::new(self.context, prototype.name()))
            .function_type(TypeAttribute::new(
                FunctionType::new(self.context, &argument_types, &result_types).into(),
            ))
            .sym_visibility(
                (prototype.name() != "main").then(|| StringAttribute::new(self.context, "private")),
            )
            .build())
    }

    fn variable_type(&self, variable: &VariableDeclaration) -> Result<Type<'c>, Error> {
        match variable.r#type() {
            ast::Type::Struct(name) => self
                .structs
                .get(name.as_str())
                .map(|(r#type, _)| (*r#type).into())
                .ok_or_else(|| error(variable.location(), format!("unknown struct type '{name}'"))),
            ast::Type::Tensor(shape) => self.tensor_type(shape, variable.location()),
        }
    }

    fn tensor_type(&self, shape: &[u64], location: crate::Location) -> Result<Type<'c>, Error> {
        Ok(if shape.is_empty() {
            self.unranked_tensor_type()
        } else {
            self.ranked_tensor_type(shape, location)?.into()
        })
    }

    fn ranked_tensor_type(
        &self,
        shape: &[u64],
        location: crate::Location,
    ) -> Result<RankedTensorType<'c>, Error> {
        if shape
            .iter()
            .try_fold(size_of::<f64>() as i64, |size, &dimension| {
                size.checked_mul(i64::try_from(dimension).ok()?)
            })
            .is_none()
        {
            return Err(error(location, "tensor shape is too large".into()));
        }

        Ok(RankedTensorType::new(
            shape,
            Type::float64(self.context),
            None,
        ))
    }

    fn unranked_tensor_type(&self) -> Type<'c> {
        Type::parse(self.context, "tensor<*xf64>").expect("valid unranked tensor type")
    }

    fn location(&self, location: crate::Location) -> Location<'c> {
        Location::new(self.context, self.path, location.line(), location.column())
    }
}

struct FunctionGenerator<'c, 'a, 'b> {
    module: &'b ModuleGenerator<'c, 'a>,
    block: &'b Block<'c>,
    scopes: Vec<HashMap<&'a str, (Value<'c, 'b>, &'a VariableDeclaration)>>,
}

impl<'c, 'a, 'b> FunctionGenerator<'c, 'a, 'b> {
    fn new(module: &'b ModuleGenerator<'c, 'a>, block: &'b Block<'c>) -> Self {
        Self {
            module,
            block,
            scopes: vec![Default::default()],
        }
    }

    fn declare(
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

    fn generate_block(&mut self, statements: &'a [Statement]) -> Result<(), Error> {
        self.scopes.push(Default::default());

        for statement in statements {
            match statement {
                Statement::Expression(expression) => {
                    self.generate_expression(expression)?;
                }
                Statement::Print(print) => {
                    let argument = self.generate_expression(print.argument())?;

                    self.block.append_operation(
                        toy::print(
                            self.module.context,
                            argument,
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
        let values = r#return
            .value()
            .map(|value| self.generate_expression(value))
            .transpose()?;

        self.block.append_operation(
            toy::r#return(
                self.module.context,
                values.as_slice(),
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
        let mut value = self.generate_expression(definition.initializer())?;

        match declaration.r#type() {
            ast::Type::Struct(_) => {
                let r#type = self.module.variable_type(declaration)?;

                if r#type != value.r#type() {
                    return Err(error(
                        declaration.location(),
                        format!(
                            "struct type of initializer is different than the variable declaration. Got {}, but expected {type}",
                            value.r#type()
                        ),
                    ));
                }
            }
            // A reshape operation is emitted for a declaration with a shape. It
            // gets optimized out later as needed.
            ast::Type::Tensor(shape) if !shape.is_empty() => {
                value = self
                    .block
                    .append_operation(
                        toy::reshape(
                            self.module.context,
                            self.module.tensor_type(shape, declaration.location())?,
                            value,
                            self.module.location(declaration.location()),
                        )
                        .into(),
                    )
                    .result(0)?
                    .into();
            }
            ast::Type::Tensor(_) => {}
        }

        self.declare(declaration, value)
    }

    fn generate_expression(&mut self, expression: &'a Expression) -> Result<Value<'c, 'b>, Error> {
        match expression {
            Expression::Binary(binary) => self.generate_binary(binary),
            Expression::Call(call) => self.generate_call(call),
            Expression::Literal(literal) => {
                let location = self.module.location(literal.location());

                Ok(self
                    .block
                    .append_operation(
                        toy::constant(
                            self.module.context,
                            self.module
                                .tensor_type(literal.dimensions(), literal.location())?,
                            self.literal_attribute(literal)?,
                            location,
                        )
                        .into(),
                    )
                    .result(0)?
                    .into())
            }
            Expression::Number(number) => {
                let r#type = RankedTensorType::new(&[], Type::float64(self.module.context), None);

                Ok(self
                    .block
                    .append_operation(
                        toy::constant(
                            self.module.context,
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
                            self.module.context,
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
                        self.module.context,
                        TupleType::try_from(lhs.r#type())?.r#type(index)?,
                        lhs,
                        IntegerAttribute::new(
                            IntegerType::new(self.module.context, 64).into(),
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
                    toy::add(self.module.context, output, lhs, rhs, location).into()
                }
                BinaryOperator::Multiply => {
                    toy::mul(self.module.context, output, lhs, rhs, location).into()
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

    // Returns the struct definition of a type of an expression if it is a
    // struct.
    fn struct_definition(&self, expression: &Expression) -> Option<&'a ast::Struct> {
        let name = match expression {
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
        };

        self.module
            .structs
            .get(name.as_str())
            .map(|(_, r#struct)| *r#struct)
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
                    "toy.transpose does not accept multiple arguments".into(),
                ));
            };

            return Ok(self
                .block
                .append_operation(
                    toy::transpose(
                        self.module.context,
                        self.module.unranked_tensor_type(),
                        *argument,
                        location,
                    )
                    .into(),
                )
                .result(0)?
                .into());
        }

        let Some(function_type) = self.module.functions.get(call.callee()) else {
            return Err(error(
                call.location(),
                format!("no defined function found for '{}'", call.callee()),
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
                    self.module.context,
                    function_type.result(0)?,
                    &arguments,
                    FlatSymbolRefAttribute::new(self.module.context, call.callee()),
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
        let mut attributes = vec![];
        let mut types = vec![];

        for value in literal.values() {
            let (attribute, r#type) = match value {
                Expression::Literal(literal) => (
                    self.literal_attribute(literal)?.into(),
                    self.module.unranked_tensor_type(),
                ),
                Expression::Number(number) => (
                    DenseElementsAttribute::f64_values(
                        RankedTensorType::new(&[], Type::float64(self.module.context), None).into(),
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
            };

            attributes.push(attribute);
            types.push(r#type);
        }

        Ok((
            ArrayAttribute::new(self.module.context, &attributes),
            TupleType::new(self.module.context, &types).into(),
        ))
    }
}

const fn error(location: crate::Location, message: String) -> Error {
    Error::CodeGeneration { location, message }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{compile::create_context, parse::parse};
    use indoc::indoc;

    fn generate_source(source: &str) -> Result<String, Error> {
        let context = create_context();
        let module = generate(&context, &parse(source, "test.toy")?)?;

        Ok(module.as_operation().to_string())
    }

    #[test]
    fn generate_codegen_example() {
        insta::assert_snapshot!(generate_source(include_str!("../examples/codegen.toy")).unwrap());
    }

    #[test]
    fn generate_scalar_example() {
        insta::assert_snapshot!(generate_source(include_str!("../examples/scalar.toy")).unwrap());
    }

    #[test]
    fn generate_struct_example() {
        insta::assert_snapshot!(
            generate_source(include_str!("../examples/struct_codegen.toy")).unwrap()
        );
    }

    #[test]
    fn generate_addition_and_nested_struct() {
        insta::assert_snapshot!(
            generate_source(indoc!(
                "
                struct Inner {
                  var a;
                }

                struct Outer {
                  Inner inner;
                  var b;
                }

                def main() {
                  Outer x = {{[1, 2]}, [3, 4]};
                  print(x.inner.a + x.b);
                  return;
                }
                "
            ))
            .unwrap()
        );
    }

    #[test]
    fn generate_largest_tensor_shape() {
        assert!(generate_source("def main() { var a<1152921504606846975> = 1; }").is_ok());
    }

    #[test]
    fn fail_to_generate_unknown_variable() {
        assert!(matches!(
            generate_source("def main() { print(a); }"),
            Err(Error::CodeGeneration { message, .. }) if message == "unknown variable 'a'"
        ));
    }

    #[test]
    fn fail_to_generate_unknown_function() {
        assert!(matches!(
            generate_source("def main() { foo(1); }"),
            Err(Error::CodeGeneration { message, .. })
                if message == "no defined function found for 'foo'"
        ));
    }

    #[test]
    fn fail_to_generate_duplicate_variable() {
        assert!(matches!(
            generate_source("def main() { var a = 1; var a = 2; }"),
            Err(Error::CodeGeneration { message, .. })
                if message == "variable 'a' is already declared"
        ));
    }

    #[test]
    fn fail_to_generate_subtraction() {
        assert!(matches!(
            generate_source("def main() { var a = 1 - 2; }"),
            Err(Error::CodeGeneration { message, .. })
                if message == "invalid binary operator '-'"
        ));
    }

    #[test]
    fn fail_to_generate_invalid_struct_access() {
        assert!(matches!(
            generate_source("def main() { var a = 1; var b = a.c; }"),
            Err(Error::CodeGeneration { message, .. })
                if message == "invalid access into struct expression"
        ));
    }

    #[test]
    fn fail_to_generate_mismatched_struct_initializer() {
        assert!(matches!(
            generate_source("struct Foo { var a; var b; } def main() { Foo x = {1}; }"),
            Err(Error::CodeGeneration { message, .. })
                if message.starts_with("struct type of initializer is different")
        ));
    }

    #[test]
    fn fail_to_generate_unknown_struct() {
        assert!(matches!(
            generate_source("def main() { Foo x = {1}; }"),
            Err(Error::CodeGeneration { message, .. }) if message == "unknown struct type 'Foo'"
        ));
    }

    #[test]
    fn fail_to_generate_dimension_above_maximum() {
        assert!(matches!(
            generate_source("def main() { var a<0, 9223372036854775808> = 1; }"),
            Err(Error::CodeGeneration { message, .. }) if message == "tensor shape is too large"
        ));
    }

    #[test]
    fn fail_to_generate_tensor_size_above_maximum() {
        assert!(matches!(
            generate_source("def main() { var a<1152921504606846976> = 1; }"),
            Err(Error::CodeGeneration { message, .. }) if message == "tensor shape is too large"
        ));
    }

    #[test]
    fn fail_to_generate_wrapping_tensor_shape() {
        assert!(matches!(
            generate_source("def main() { var a<3, 6148914691236517206> = [1, 2]; }"),
            Err(Error::CodeGeneration { message, .. }) if message == "tensor shape is too large"
        ));
    }

    #[test]
    fn fail_to_generate_too_large_literal() {
        let location = crate::Location::new(1, 1);

        assert!(matches!(
            generate(
                &create_context(),
                &ast::Module::new(
                    "test.toy".into(),
                    vec![Record::Function(Function::new(
                        ast::Prototype::new("main".into(), vec![], location),
                        vec![Statement::Expression(Expression::StructLiteral(
                            StructLiteral::new(
                                vec![Expression::Literal(Literal::new(
                                    vec![Expression::Number(ast::Number::new(1.0, location))],
                                    vec![1 << 32, 1 << 32],
                                    location,
                                ))],
                                location,
                            )
                        ))],
                    ))],
                ),
            ),
            Err(Error::CodeGeneration { message, .. }) if message == "tensor shape is too large"
        ));
    }
}
