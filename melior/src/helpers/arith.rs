use super::builtin::BuiltinBlockExt;
use crate::{
    Context, Error,
    dialect::{
        arith::{
            CmpiPredicate, addi, andi, cmpi, divsi, divui, extsi, extui, muli, ori, shli, shrsi,
            shrui, subi, trunci, xori,
        },
        ods,
    },
    ir::{Attribute, Block, Location, Type, Value, r#type::IntegerType},
};
use core::fmt::Display;

macro_rules! binary_operation_declaration {
    ($name:ident, $documentation:literal) => {
        #[doc=$documentation]
        fn $name(
            &self,
            lhs: Value<'c, '_>,
            rhs: Value<'c, '_>,
            location: Location<'c>,
        ) -> Result<Value<'c, '_>, Error>;
    };
}

macro_rules! binary_operation {
    ($name:ident) => {
        #[inline]
        fn $name(
            &self,
            lhs: Value<'c, '_>,
            rhs: Value<'c, '_>,
            location: Location<'c>,
        ) -> Result<Value<'c, '_>, Error> {
            self.append_op_result($name(lhs, rhs, location))
        }
    };
}

/// A block extension for an `arith` dialect.
pub trait ArithBlockExt<'c>: BuiltinBlockExt<'c> {
    binary_operation_declaration!(addi, "Creates an `arith.addi` operation.");
    binary_operation_declaration!(andi, "Creates an `arith.andi` operation.");
    binary_operation_declaration!(divsi, "Creates an `arith.divsi` operation.");
    binary_operation_declaration!(divui, "Creates an `arith.divui` operation.");
    binary_operation_declaration!(muli, "Creates an `arith.muli` operation.");
    binary_operation_declaration!(ori, "Creates an `arith.ori` operation.");
    binary_operation_declaration!(shli, "Creates an `arith.shli` operation.");
    binary_operation_declaration!(shrsi, "Creates an `arith.shrsi` operation.");
    binary_operation_declaration!(shrui, "Creates an `arith.shrui` operation.");
    binary_operation_declaration!(subi, "Creates an `arith.subi` operation.");
    binary_operation_declaration!(xori, "Creates an `arith.xori` operation.");

    /// Creates an `arith.cmpi` operation.
    fn cmpi(
        &self,
        context: &'c Context,
        pred: CmpiPredicate,
        lhs: Value<'c, '_>,
        rhs: Value<'c, '_>,
        location: Location<'c>,
    ) -> Result<Value<'c, '_>, Error>;

    /// Creates an `arith.extui` operation.
    fn extui(
        &self,
        lhs: Value<'c, '_>,
        target_type: Type<'c>,
        location: Location<'c>,
    ) -> Result<Value<'c, '_>, Error>;

    /// Creates an `arith.extui` operation.
    fn extsi(
        &self,
        lhs: Value<'c, '_>,
        target_type: Type<'c>,
        location: Location<'c>,
    ) -> Result<Value<'c, '_>, Error>;

    /// Creates an `arith.extui` operation.
    fn trunci(
        &self,
        lhs: Value<'c, '_>,
        target_type: Type<'c>,
        location: Location<'c>,
    ) -> Result<Value<'c, '_>, Error>;

    /// Creates a constant of the given integer bit width.
    fn const_int(
        &self,
        context: &'c Context,
        location: Location<'c>,
        value: impl Display,
        bits: u32,
    ) -> Result<Value<'c, '_>, Error>;

    /// Creates a constant of the given integer type.
    fn const_int_from_type(
        &self,
        context: &'c Context,
        location: Location<'c>,
        value: impl Display,
        r#type: Type<'c>,
    ) -> Result<Value<'c, '_>, Error>;
}

impl<'c> ArithBlockExt<'c> for Block<'c> {
    binary_operation!(addi);
    binary_operation!(andi);
    binary_operation!(divsi);
    binary_operation!(divui);
    binary_operation!(muli);
    binary_operation!(ori);
    binary_operation!(shli);
    binary_operation!(shrsi);
    binary_operation!(shrui);
    binary_operation!(subi);
    binary_operation!(xori);

    #[inline]
    fn cmpi(
        &self,
        context: &'c Context,
        predicate: CmpiPredicate,
        lhs: Value<'c, '_>,
        rhs: Value<'c, '_>,
        location: Location<'c>,
    ) -> Result<Value<'c, '_>, Error> {
        self.append_op_result(cmpi(context, predicate, lhs, rhs, location))
    }

    #[inline]
    fn extsi(
        &self,
        value: Value<'c, '_>,
        target_type: Type<'c>,
        location: Location<'c>,
    ) -> Result<Value<'c, '_>, Error> {
        self.append_op_result(extsi(value, target_type, location))
    }

    #[inline]
    fn extui(
        &self,
        value: Value<'c, '_>,
        target_type: Type<'c>,
        location: Location<'c>,
    ) -> Result<Value<'c, '_>, Error> {
        self.append_op_result(extui(value, target_type, location))
    }

    #[inline]
    fn trunci(
        &self,
        value: Value<'c, '_>,
        target_type: Type<'c>,
        location: Location<'c>,
    ) -> Result<Value<'c, '_>, Error> {
        self.append_op_result(trunci(value, target_type, location))
    }

    #[inline]
    fn const_int(
        &self,
        context: &'c Context,
        location: Location<'c>,
        value: impl Display,
        bits: u32,
    ) -> Result<Value<'c, '_>, Error> {
        self.const_int_from_type(
            context,
            location,
            value,
            IntegerType::new(context, bits).into(),
        )
    }

    #[inline]
    fn const_int_from_type(
        &self,
        context: &'c Context,
        location: Location<'c>,
        value: impl Display,
        r#type: Type<'c>,
    ) -> Result<Value<'c, '_>, Error> {
        let attribute = format!("{value} : {type}");

        self.append_op_result(
            ods::arith::constant(
                context,
                r#type,
                Attribute::parse(context, &attribute).ok_or(Error::AttributeParse(attribute))?,
                location,
            )
            .into(),
        )
    }
}
