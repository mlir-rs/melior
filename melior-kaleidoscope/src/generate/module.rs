use super::{error, function::FunctionGenerator};
use crate::{
    ast::{self, Function, Record, VariableDeclaration},
    error::Error,
    toy,
    utility::{is_operation, terminator},
};
use melior::{
    Context,
    ir::{
        Block, BlockLike, Location, Module, Region, RegionLike, Type, ValueLike,
        attribute::{StringAttribute, TypeAttribute},
        operation::OperationLike,
        r#type::{FunctionType, RankedTensorType, TupleType},
    },
};
use std::collections::HashMap;

pub struct ModuleGenerator<'c, 'a> {
    context: &'c Context,
    path: &'a str,
    functions: HashMap<&'a str, FunctionType<'c>>,
    structs: HashMap<&'a str, (TupleType<'c>, &'a ast::Struct)>,
}

impl<'c, 'a> ModuleGenerator<'c, 'a> {
    pub fn new(context: &'c Context, path: &'a str) -> Self {
        Self {
            context,
            path,
            functions: Default::default(),
            structs: Default::default(),
        }
    }

    pub const fn context(&self) -> &'c Context {
        self.context
    }

    pub fn function_type(&self, name: &str) -> Option<FunctionType<'c>> {
        self.functions.get(name).copied()
    }

    pub fn struct_definition(&self, name: &str) -> Option<&'a ast::Struct> {
        self.structs.get(name).map(|(_, r#struct)| *r#struct)
    }

    pub fn generate(mut self, module: &'a ast::Module) -> Result<Module<'c>, Error> {
        let mlir_module = Module::new(Location::unknown(self.context));

        for record in module.records() {
            match record {
                Record::Function(function) => {
                    mlir_module
                        .body()
                        .append_operation(self.generate_function(function)?.into());
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
                format!("struct type '{}' is already defined", r#struct.name()),
            ));
        }

        self.structs.insert(
            r#struct.name(),
            (
                TupleType::new(
                    self.context,
                    &r#struct
                        .variables()
                        .iter()
                        .map(|variable| {
                            if matches!(
                                variable.r#type(),
                                ast::Type::Tensor(shape) if !shape.is_empty()
                            ) {
                                return Err(error(
                                    variable.location(),
                                    "variables within a struct definition must not have shapes"
                                        .into(),
                                ));
                            }

                            self.variable_type(variable)
                        })
                        .collect::<Result<Vec<_>, _>>()?,
                ),
                r#struct,
            ),
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
        let function_type = FunctionType::new(self.context, &argument_types, &result_types);

        self.functions.insert(prototype.name(), function_type);

        let region = Region::new();
        region.append_block(block);

        // Functions other than the main one are private so that they are
        // removed after inlining.
        Ok(toy::FuncOperation::builder(self.context, location)
            .body(region)
            .sym_name(StringAttribute::new(self.context, prototype.name()))
            .function_type(TypeAttribute::new(function_type.into()))
            .sym_visibility(
                (prototype.name() != "main").then(|| StringAttribute::new(self.context, "private")),
            )
            .build())
    }

    pub fn variable_type(&self, variable: &VariableDeclaration) -> Result<Type<'c>, Error> {
        match variable.r#type() {
            ast::Type::Struct(name) => self
                .structs
                .get(name.as_str())
                .map(|(r#type, _)| (*r#type).into())
                .ok_or_else(|| error(variable.location(), format!("unknown struct type '{name}'"))),
            ast::Type::Tensor(shape) => self.tensor_type(shape, variable.location()),
        }
    }

    pub fn tensor_type(&self, shape: &[u64], location: crate::Location) -> Result<Type<'c>, Error> {
        Ok(if shape.is_empty() {
            self.unranked_tensor_type()
        } else {
            self.ranked_tensor_type(shape, location)?.into()
        })
    }

    pub fn ranked_tensor_type(
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

    pub fn unranked_tensor_type(&self) -> Type<'c> {
        Type::parse(self.context, "tensor<*xf64>").expect("valid unranked tensor type")
    }

    pub fn location(&self, location: crate::Location) -> Location<'c> {
        Location::new(self.context, self.path, location.line(), location.column())
    }
}
