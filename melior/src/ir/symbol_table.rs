use super::{
    Attribute, AttributeLike, BlockLike, Operation, OperationRef, RegionLike,
    attribute::StringAttribute, operation::OperationLike,
};
use crate::{Error, StringRef};
use mlir_sys::{
    MlirSymbolTable, mlirOperationGetAttributeByName, mlirSymbolTableCreate,
    mlirSymbolTableDestroy, mlirSymbolTableErase, mlirSymbolTableGetSymbolAttributeName,
    mlirSymbolTableGetVisibilityAttributeName, mlirSymbolTableInsert, mlirSymbolTableLookup,
};

/// A symbol table.
///
/// A symbol table holds symbols found in its operation on its creation and
/// symbols inserted through it. It does not track symbols added, renamed, or
/// removed in other ways.
pub struct SymbolTable<'c, 'a> {
    raw: MlirSymbolTable,
    operation: OperationRef<'c, 'a>,
}

impl<'c, 'a> SymbolTable<'c, 'a> {
    /// Creates a symbol table for an operation.
    pub fn new(operation: impl OperationLike<'c, 'a>) -> Result<Self, Error> {
        (operation.region_count() == 1
            && operation
                .first_region()
                .and_then(RegionLike::first_block)
                .is_some_and(|block| block.next_in_region().is_none()))
        .then(|| unsafe { mlirSymbolTableCreate(operation.to_raw()) })
        .filter(|raw| !raw.ptr.is_null())
        .map(|raw| Self {
            raw,
            operation: unsafe { OperationRef::from_raw(operation.to_raw()) },
        })
        .ok_or_else(|| Error::SymbolTableExpected(operation.to_string()))
    }

    /// Returns an operation.
    pub const fn operation(&self) -> OperationRef<'c, 'a> {
        self.operation
    }

    /// Looks up a symbol.
    pub fn lookup(&self, name: &str) -> Option<OperationRef<'c, 'a>> {
        unsafe {
            OperationRef::from_option_raw(mlirSymbolTableLookup(
                self.raw,
                StringRef::new(name).to_raw(),
            ))
        }
    }

    /// Inserts a symbol and returns its name.
    ///
    /// The symbol is appended to the operation of the symbol table, before a
    /// terminator if any, and renamed if its name is taken already.
    pub fn insert(&mut self, symbol: Operation<'c>) -> Result<StringAttribute<'c>, Error> {
        if is_symbol(&symbol) {
            StringAttribute::try_from(unsafe {
                Attribute::from_raw(mlirSymbolTableInsert(self.raw, symbol.into_raw()))
            })
        } else {
            Err(Error::SymbolExpected(symbol.to_string()))
        }
    }

    /// Erases a symbol.
    ///
    /// # Safety
    ///
    /// References to the symbol and IR nested in it must not be used after the
    /// call, including ones held by walks in progress and ones looked up in
    /// other symbol tables later. Values defined in the symbol must have no
    /// uses outside of it.
    // TODO Make this safe when mutable operation references are exclusive.
    pub unsafe fn erase(&mut self, symbol: OperationRef<'c, '_>) -> Result<(), Error> {
        if !is_symbol(symbol) || symbol.parent_operation() != Some(self.operation) {
            return Err(Error::SymbolExpected(symbol.to_string()));
        }

        unsafe { mlirSymbolTableErase(self.raw, symbol.to_raw()) };

        Ok(())
    }

    /// Returns a name of attributes that hold symbol names.
    pub fn symbol_attribute_name() -> StringRef<'static> {
        unsafe { StringRef::from_raw(mlirSymbolTableGetSymbolAttributeName()) }
    }

    /// Returns a name of attributes that hold symbol visibility.
    pub fn visibility_attribute_name() -> StringRef<'static> {
        unsafe { StringRef::from_raw(mlirSymbolTableGetVisibilityAttributeName()) }
    }

    /// Converts a symbol table into a raw object.
    pub const fn to_raw(&self) -> MlirSymbolTable {
        self.raw
    }
}

impl Drop for SymbolTable<'_, '_> {
    fn drop(&mut self) {
        unsafe { mlirSymbolTableDestroy(self.raw) };
    }
}

fn is_symbol<'c: 'a, 'a>(operation: impl OperationLike<'c, 'a>) -> bool {
    unsafe {
        Attribute::from_option_raw(mlirOperationGetAttributeByName(
            operation.to_raw(),
            SymbolTable::symbol_attribute_name().to_raw(),
        ))
    }
    .is_some_and(|attribute| attribute.is_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Context,
        dialect::func,
        ir::{
            Block, Identifier, Location, Module, Region, attribute::TypeAttribute,
            operation::OperationBuilder, r#type::FunctionType,
        },
        test::create_test_context,
    };
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    fn create_function<'c>(context: &'c Context, name: &str) -> Operation<'c> {
        func::func(
            context,
            StringAttribute::new(context, name),
            TypeAttribute::new(FunctionType::new(context, &[], &[]).into()),
            Region::new(),
            &[(
                Identifier::new(context, "sym_visibility"),
                StringAttribute::new(context, "private").into(),
            )],
            Location::unknown(context),
        )
    }

    fn create_module<'c>(context: &'c Context, regions: Vec<Region<'c>>) -> Operation<'c> {
        OperationBuilder::new("builtin.module", Location::unknown(context))
            .add_regions_vec(regions)
            .build()
            .unwrap()
    }

    fn create_region<'c>(block_count: usize) -> Region<'c> {
        let region = Region::new();

        for _ in 0..block_count {
            region.append_block(Block::new(&[], &[]).unwrap());
        }

        region
    }

    #[test]
    fn new() {
        let context = create_test_context();
        let module = Module::new(Location::unknown(&context));

        assert!(SymbolTable::new(module.as_operation()).is_ok());
    }

    #[test]
    fn new_with_owned_operation() {
        let context = create_test_context();

        assert!(SymbolTable::new(&create_module(&context, vec![create_region(1)])).is_ok());
    }

    #[test]
    fn new_with_nested_symbol_table() {
        let context = create_test_context();
        let module = Module::parse(
            &context,
            indoc!(
                "
                module {
                    module @foo {
                        func.func private @bar()
                    }
                }
                "
            ),
        )
        .unwrap();
        let table = SymbolTable::new(module.body().first_operation().unwrap()).unwrap();

        assert!(table.lookup("foo").is_none());
        assert!(table.lookup("bar").is_some());
    }

    #[test]
    fn fail_to_create_with_non_symbol_table() {
        let context = create_test_context();
        let module = Module::parse(&context, "func.func @foo() { return }").unwrap();
        let function = module.body().first_operation().unwrap();

        assert_eq!(
            SymbolTable::new(function).err(),
            Some(Error::SymbolTableExpected(function.to_string()))
        );
    }

    #[test]
    fn fail_to_create_without_region() {
        let context = create_test_context();
        let module = create_module(&context, vec![]);

        assert_eq!(
            SymbolTable::new(&module).err(),
            Some(Error::SymbolTableExpected(module.to_string()))
        );
    }

    #[test]
    fn fail_to_create_with_multiple_regions() {
        let context = create_test_context();
        let module = create_module(&context, vec![create_region(1), create_region(1)]);

        assert_eq!(
            SymbolTable::new(&module).err(),
            Some(Error::SymbolTableExpected(module.to_string()))
        );
    }

    #[test]
    fn fail_to_create_without_block() {
        let context = create_test_context();
        let module = create_module(&context, vec![create_region(0)]);

        assert_eq!(
            SymbolTable::new(&module).err(),
            Some(Error::SymbolTableExpected(module.to_string()))
        );
    }

    #[test]
    fn fail_to_create_with_multiple_blocks() {
        let context = create_test_context();
        let module = create_module(&context, vec![create_region(2)]);

        assert_eq!(
            SymbolTable::new(&module).err(),
            Some(Error::SymbolTableExpected(module.to_string()))
        );
    }

    #[test]
    fn operation() {
        let context = create_test_context();
        let module = Module::new(Location::unknown(&context));

        assert_eq!(
            SymbolTable::new(module.as_operation()).unwrap().operation(),
            module.as_operation()
        );
    }

    #[test]
    fn lookup() {
        let context = create_test_context();
        let module = Module::new(Location::unknown(&context));
        let function = module
            .body()
            .append_operation(create_function(&context, "foo"));

        assert_eq!(
            SymbolTable::new(module.as_operation())
                .unwrap()
                .lookup("foo"),
            Some(function)
        );
    }

    #[test]
    fn lookup_none() {
        let context = create_test_context();
        let module = Module::new(Location::unknown(&context));

        module
            .body()
            .append_operation(create_function(&context, "foo"));

        assert_eq!(
            SymbolTable::new(module.as_operation())
                .unwrap()
                .lookup("bar"),
            None
        );
    }

    #[test]
    fn insert() {
        let context = create_test_context();
        let module = Module::new(Location::unknown(&context));
        let mut table = SymbolTable::new(module.as_operation()).unwrap();

        assert_eq!(
            table
                .insert(create_function(&context, "foo"))
                .unwrap()
                .value(),
            Ok("foo")
        );
        assert!(table.lookup("foo").is_some());
        assert_eq!(table.lookup("foo"), module.body().first_operation());
        assert!(module.as_operation().verify());
    }

    #[test]
    fn insert_with_name_conflict() {
        let context = create_test_context();
        let module = Module::new(Location::unknown(&context));
        let function = module
            .body()
            .append_operation(create_function(&context, "foo"));
        let mut table = SymbolTable::new(module.as_operation()).unwrap();

        assert_eq!(
            table
                .insert(create_function(&context, "foo"))
                .unwrap()
                .value(),
            Ok("foo_0")
        );
        assert_eq!(table.lookup("foo"), Some(function));
        assert_eq!(table.lookup("foo_0"), function.next_in_block());
        assert!(module.as_operation().verify());
    }

    #[test]
    fn insert_before_terminator() {
        let context = create_test_context();
        let module = Module::new(Location::unknown(&context));
        let terminator = module
            .body()
            .append_operation(func::r#return(&[], Location::unknown(&context)));
        let mut table = SymbolTable::new(module.as_operation()).unwrap();

        table.insert(create_function(&context, "foo")).unwrap();

        assert_eq!(table.lookup("foo"), module.body().first_operation());
        assert_eq!(
            table.lookup("foo").unwrap().next_in_block(),
            Some(terminator)
        );
    }

    #[test]
    fn fail_to_insert_without_name() {
        let context = create_test_context();
        context.set_allow_unregistered_dialects(true);
        let module = Module::new(Location::unknown(&context));
        let operation = OperationBuilder::new("foo", Location::unknown(&context))
            .build()
            .unwrap();

        assert_eq!(
            SymbolTable::new(module.as_operation())
                .unwrap()
                .insert(operation.clone())
                .err(),
            Some(Error::SymbolExpected(operation.to_string()))
        );
        assert_eq!(module.body().first_operation(), None);
    }

    #[test]
    fn fail_to_insert_with_non_string_name() {
        let context = create_test_context();
        context.set_allow_unregistered_dialects(true);
        let module = Module::new(Location::unknown(&context));
        let operation = OperationBuilder::new("foo", Location::unknown(&context))
            .add_attributes(&[(
                Identifier::new(&context, "sym_name"),
                Attribute::unit(&context),
            )])
            .build()
            .unwrap();

        assert_eq!(
            SymbolTable::new(module.as_operation())
                .unwrap()
                .insert(operation.clone())
                .err(),
            Some(Error::SymbolExpected(operation.to_string()))
        );
        assert_eq!(module.body().first_operation(), None);
    }

    #[test]
    fn erase() {
        let context = create_test_context();
        let module = Module::new(Location::unknown(&context));
        let function = module
            .body()
            .append_operation(create_function(&context, "foo"));
        let mut table = SymbolTable::new(module.as_operation()).unwrap();

        assert_eq!(unsafe { table.erase(function) }, Ok(()));
        assert_eq!(table.lookup("foo"), None);
        assert_eq!(module.body().first_operation(), None);
    }

    #[test]
    fn fail_to_erase_non_symbol() {
        let context = create_test_context();
        context.set_allow_unregistered_dialects(true);
        let module = Module::new(Location::unknown(&context));
        let operation = module.body().append_operation(
            OperationBuilder::new("foo", Location::unknown(&context))
                .build()
                .unwrap(),
        );

        assert_eq!(
            unsafe {
                SymbolTable::new(module.as_operation())
                    .unwrap()
                    .erase(operation)
            },
            Err(Error::SymbolExpected(operation.to_string()))
        );
        assert_eq!(module.body().first_operation(), Some(operation));
    }

    #[test]
    fn fail_to_erase_symbol_in_another_symbol_table() {
        let context = create_test_context();
        let module = Module::new(Location::unknown(&context));
        let other_module = Module::new(Location::unknown(&context));
        let function = other_module
            .body()
            .append_operation(create_function(&context, "foo"));

        assert_eq!(
            unsafe {
                SymbolTable::new(module.as_operation())
                    .unwrap()
                    .erase(function)
            },
            Err(Error::SymbolExpected(function.to_string()))
        );
        assert_eq!(other_module.body().first_operation(), Some(function));
    }

    #[test]
    fn fail_to_erase_detached_symbol() {
        let context = create_test_context();
        let module = Module::new(Location::unknown(&context));
        let function = create_function(&context, "foo");

        assert_eq!(
            unsafe {
                SymbolTable::new(module.as_operation())
                    .unwrap()
                    .erase(OperationRef::from_raw(function.to_raw()))
            },
            Err(Error::SymbolExpected(function.to_string()))
        );
    }

    #[test]
    fn symbol_attribute_name() {
        assert_eq!(
            SymbolTable::symbol_attribute_name().as_str(),
            Ok("sym_name")
        );
    }

    #[test]
    fn visibility_attribute_name() {
        assert_eq!(
            SymbolTable::visibility_attribute_name().as_str(),
            Ok("sym_visibility")
        );
    }
}
