use super::error::{Error, OdsError};
use once_cell::sync::Lazy;
use std::collections::HashMap;
use tblgen::{
    error::{TableGenError, WithLocation},
    record::Record,
};

macro_rules! prefixed_string {
    ($prefix:literal, $name:ident) => {
        concat!($prefix, stringify!($name))
    };
}

macro_rules! mlir_attribute {
    ($name:ident) => {
        prefixed_string!("::mlir::", $name)
    };
}

macro_rules! melior_attribute {
    ($name:ident) => {
        prefixed_string!("::melior::ir::attribute::", $name)
    };
}

static ATTRIBUTE_TYPES: Lazy<HashMap<&'static str, &'static str>> = Lazy::new(|| {
    let mut map = HashMap::new();

    macro_rules! initialize_attributes {
        ($($mlir:ident => $melior:ident),* $(,)*) => {
            $(
                map.insert(
                    mlir_attribute!($mlir),
                    melior_attribute!($melior),
                );
            )*
        };
    }

    initialize_attributes!(
        ArrayAttr => ArrayAttribute,
        Attribute => Attribute,
        DenseElementsAttr => DenseElementsAttribute,
        DenseI32ArrayAttr => DenseI32ArrayAttribute,
        FlatSymbolRefAttr => FlatSymbolRefAttribute,
        FloatAttr => FloatAttribute,
        IntegerAttr => IntegerAttribute,
        StringAttr => StringAttribute,
        TypeAttr => TypeAttribute,
    );

    map
});

#[derive(Debug, Clone, Copy)]
pub struct RegionConstraint<'a>(Record<'a>);

impl<'a> RegionConstraint<'a> {
    pub fn new(record: Record<'a>) -> Self {
        Self(record)
    }

    pub fn is_variadic(&self) -> bool {
        self.0.subclass_of("VariadicRegion")
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SuccessorConstraint<'a>(Record<'a>);

impl<'a> SuccessorConstraint<'a> {
    pub fn new(record: Record<'a>) -> Self {
        Self(record)
    }

    pub fn is_variadic(&self) -> bool {
        self.0.subclass_of("VariadicSuccessor")
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TypeConstraint<'a>(Record<'a>);

impl<'a> TypeConstraint<'a> {
    pub fn new(record: Record<'a>) -> Self {
        Self(record)
    }

    pub fn is_optional(&self) -> bool {
        self.0.subclass_of("Optional")
    }

    pub fn is_variadic(&self) -> bool {
        self.0.subclass_of("Variadic")
    }

    #[allow(unused)]
    pub fn is_variadic_of_variadic(&self) -> bool {
        self.0.subclass_of("VariadicOfVariadic")
    }

    pub fn has_variable_length(&self) -> bool {
        self.is_variadic() || self.is_optional()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct AttributeConstraint<'a>(Record<'a>);

impl<'a> AttributeConstraint<'a> {
    pub fn new(record: Record<'a>) -> Self {
        Self(record)
    }

    #[allow(unused)]
    pub fn is_derived(&self) -> bool {
        self.0.subclass_of("DerivedAttr")
    }

    #[allow(unused)]
    pub fn is_type(&self) -> bool {
        self.0.subclass_of("TypeAttrBase")
    }

    #[allow(unused)]
    pub fn is_symbol_ref(&self) -> bool {
        self.0.name() == Ok("SymbolRefAttr")
            || self.0.name() == Ok("FlatSymbolRefAttr")
            || self.0.subclass_of("SymbolRefAttr")
            || self.0.subclass_of("FlatSymbolRefAttr")
    }

    #[allow(unused)]
    pub fn is_enum(&self) -> bool {
        self.0.subclass_of("EnumAttrInfo")
    }

    pub fn is_optional(&self) -> Result<bool, Error> {
        Ok(self.0.bit_value("isOptional")?)
    }

    pub fn storage_type(&self) -> Result<&'static str, Error> {
        Ok(ATTRIBUTE_TYPES
            .get(self.0.string_value("storageType")?.as_str().trim())
            .copied()
            .unwrap_or(melior_attribute!(Attribute)))
    }

    pub fn is_unit(&self) -> Result<bool, Error> {
        Ok(self.0.string_value("storageType")? == mlir_attribute!(UnitAttr))
    }

    pub fn has_default_value(&self) -> Result<bool, Error> {
        Ok(match self.0.string_value("defaultValue") {
            Ok(value) => !value.is_empty(),
            Err(error) => {
                // `defaultValue` can be uninitialized.
                if !matches!(error.error(), TableGenError::InitConversion { .. }) {
                    return Err(error.into());
                }

                false
            }
        })
    }
}

#[derive(Debug, Clone)]
enum TraitKind {
    Native {
        name: String,
        #[allow(unused)]
        structural: bool,
    },
    Pred {},
    Internal {
        name: String,
    },
    Interface {
        name: String,
    },
}

#[derive(Debug, Clone)]
pub struct Trait {
    kind: TraitKind,
}

impl Trait {
    pub fn new(def: Record) -> Result<Self, Error> {
        Ok(Self {
            kind: if def.subclass_of("PredTrait") {
                TraitKind::Pred {}
            } else if def.subclass_of("InterfaceTrait") {
                TraitKind::Interface {
                    name: Self::name(def)?,
                }
            } else if def.subclass_of("NativeTrait") {
                TraitKind::Native {
                    name: Self::name(def)?,
                    structural: def.subclass_of("StructuralOpTrait"),
                }
            } else if def.subclass_of("GenInternalTrait") {
                TraitKind::Internal {
                    name: def.string_value("trait")?,
                }
            } else {
                return Err(OdsError::InvalidTrait.with_location(def).into());
            },
        })
    }

    pub fn has_name(&self, expected_name: &str) -> bool {
        match &self.kind {
            TraitKind::Native { name, .. }
            | TraitKind::Internal { name }
            | TraitKind::Interface { name } => expected_name == name,
            TraitKind::Pred {} => false,
        }
    }

    fn name(def: Record) -> Result<String, Error> {
        let r#trait = def.string_value("trait")?;
        let namespace = def.string_value("cppNamespace")?;

        Ok(if namespace.is_empty() {
            r#trait
        } else {
            format!("{namespace}::{trait}")
        })
    }
}
