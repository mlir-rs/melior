use super::{
    Attribute, AttributeLike, BlockLike, OperationRef, RegionLike, TypeLike,
    attribute::StringAttribute, operation::OperationLike,
};
use crate::{Error, StringRef};
use mlir_sys::{
    MlirSymbolTable, mlirSymbolTableCreate, mlirSymbolTableDestroy, mlirSymbolTableErase,
    mlirSymbolTableGetSymbolAttributeName, mlirSymbolTableGetVisibilityAttributeName,
    mlirSymbolTableInsert, mlirSymbolTableLookup,
};

/// A symbol table.
///
/// A symbol table holds symbols found in its operation on its creation and
/// symbols inserted through it. It does not track symbols added, renamed, or
/// removed in other ways.
#[derive(Debug)]
pub struct SymbolTable<'c, 'a> {
    raw: MlirSymbolTable,
    operation: OperationRef<'c, 'a>,
}

impl<'c, 'a> SymbolTable<'c, 'a> {
    /// Creates a symbol table for an operation.
    pub fn new(operation: impl OperationLike<'c, 'a>) -> Result<Self, Error> {
        if let Some(raw) = has_single_block(operation)
            .then(|| unsafe { mlirSymbolTableCreate(operation.to_raw()) })
            .filter(|raw| !raw.ptr.is_null())
        {
            Ok(Self {
                raw,
                operation: unsafe { OperationRef::from_raw(operation.to_raw()) },
            })
        } else {
            Err(Error::SymbolTableExpected(operation.to_string()))
        }
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
    /// The symbol must be in the operation of the symbol table already. It is
    /// renamed if the symbol table holds another symbol of the same name. Uses
    /// of a renamed symbol are left as they are. A symbol that the symbol table
    /// holds under an earlier name is held under both names afterwards.
    pub fn insert(&mut self, symbol: OperationRef<'c, '_>) -> Result<StringAttribute<'c>, Error> {
        self.check_symbol(symbol)?;

        StringAttribute::try_from(unsafe {
            Attribute::from_raw(mlirSymbolTableInsert(self.raw, symbol.to_raw()))
        })
    }

    /// Erases a symbol held by a symbol table.
    ///
    /// # Safety
    ///
    /// After the call, references to the symbol and IR nested in it must not be
    /// used, including ones held by walks in progress and ones looked up in
    /// this or other symbol tables later. Symbol tables created for the symbol
    /// or operations nested in it must not be used other than being dropped.
    /// Values defined in the symbol must have no uses outside of it.
    // TODO Make this safe when symbol tables borrow operations exclusively.
    pub unsafe fn erase(&mut self, symbol: OperationRef<'c, '_>) -> Result<(), Error> {
        let name = self.check_symbol(symbol)?.value()?;

        if self.lookup(name) != Some(symbol) {
            return Err(Error::SymbolNotFound(name.into()));
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

    fn check_symbol(&self, symbol: OperationRef<'c, '_>) -> Result<StringAttribute<'c>, Error> {
        let Some(name) = symbol_name(symbol) else {
            return Err(Error::SymbolExpected(symbol.to_string()));
        };

        if symbol.parent_operation() != Some(self.operation) {
            return Err(Error::SymbolNotFound(name.value()?.into()));
        }

        Ok(name)
    }
}

impl Drop for SymbolTable<'_, '_> {
    fn drop(&mut self) {
        unsafe { mlirSymbolTableDestroy(self.raw) };
    }
}

fn has_single_block<'c: 'a, 'a>(operation: impl OperationLike<'c, 'a>) -> bool {
    operation.region_count() == 1
        && operation
            .first_region()
            .and_then(RegionLike::first_block)
            .is_some_and(|block| block.next_in_region().is_none())
}

// TODO Check the symbol interface instead of the attribute on LLVM 24.
fn symbol_name<'c: 'a, 'a>(operation: impl OperationLike<'c, 'a>) -> Option<StringAttribute<'c>> {
    operation
        .attribute(SymbolTable::symbol_attribute_name().as_str().ok()?)
        .and_then(StringAttribute::try_from)
        .ok()
        .filter(|name| name.r#type().is_none())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Context, IrRewriter,
        dialect::func,
        ir::{
            Block, Identifier, Location, Module, Operation, Region,
            attribute::{FlatSymbolRefAttribute, TypeAttribute},
            operation::{OperationBuilder, OperationMutLike},
            r#type::{FunctionType, IntegerType},
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

    fn create_operation<'c>(context: &'c Context) -> Operation<'c> {
        OperationBuilder::new("foo", Location::unknown(context))
            .build()
            .unwrap()
    }

    fn create_operation_with_name<'c>(context: &'c Context, name: Attribute<'c>) -> Operation<'c> {
        OperationBuilder::new("foo", Location::unknown(context))
            .add_attributes(&[(Identifier::new(context, "sym_name"), name)])
            .build()
            .unwrap()
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

        assert_eq!(table.lookup("foo"), None);
        assert!(table.lookup("bar").is_some());
    }

    #[test]
    fn fail_to_create_with_non_symbol_table() {
        let context = create_test_context();
        let module = Module::parse(&context, "func.func @foo() { return }").unwrap();
        let function = module.body().first_operation().unwrap();

        assert_eq!(
            SymbolTable::new(function).unwrap_err(),
            Error::SymbolTableExpected(function.to_string())
        );
    }

    #[test]
    fn fail_to_create_without_region() {
        let context = create_test_context();
        let module = create_module(&context, vec![]);

        assert_eq!(
            SymbolTable::new(&module).unwrap_err(),
            Error::SymbolTableExpected(module.to_string())
        );
    }

    #[test]
    fn fail_to_create_with_multiple_regions() {
        let context = create_test_context();
        let module = create_module(&context, vec![create_region(1), create_region(1)]);

        assert_eq!(
            SymbolTable::new(&module).unwrap_err(),
            Error::SymbolTableExpected(module.to_string())
        );
    }

    #[test]
    fn fail_to_create_without_block() {
        let context = create_test_context();
        let module = create_module(&context, vec![create_region(0)]);

        assert_eq!(
            SymbolTable::new(&module).unwrap_err(),
            Error::SymbolTableExpected(module.to_string())
        );
    }

    #[test]
    fn fail_to_create_with_multiple_blocks() {
        let context = create_test_context();
        let module = create_module(&context, vec![create_region(2)]);

        assert_eq!(
            SymbolTable::new(&module).unwrap_err(),
            Error::SymbolTableExpected(module.to_string())
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
    fn lookup_none_for_symbol_appended_after_creation() {
        let context = create_test_context();
        let module = Module::new(Location::unknown(&context));
        let table = SymbolTable::new(module.as_operation()).unwrap();

        module
            .body()
            .append_operation(create_function(&context, "foo"));

        assert_eq!(table.lookup("foo"), None);
    }

    #[test]
    fn insert() {
        let context = create_test_context();
        let module = Module::new(Location::unknown(&context));
        let mut table = SymbolTable::new(module.as_operation()).unwrap();
        let function = module
            .body()
            .append_operation(create_function(&context, "foo"));

        assert_eq!(table.insert(function).unwrap().value(), Ok("foo"));
        assert_eq!(table.lookup("foo"), Some(function));
    }

    #[test]
    fn insert_with_nested_symbol_table() {
        let context = create_test_context();
        let module = Module::parse(&context, "module { module @foo {} }").unwrap();
        let nested_module = module.body().first_operation().unwrap();
        let mut table = SymbolTable::new(nested_module).unwrap();
        let function = nested_module
            .first_region()
            .unwrap()
            .first_block()
            .unwrap()
            .append_operation(create_function(&context, "bar"));

        assert_eq!(table.insert(function).unwrap().value(), Ok("bar"));
        assert_eq!(table.lookup("bar"), Some(function));
    }

    #[test]
    fn insert_with_name_conflict() {
        let context = create_test_context();
        let module = Module::new(Location::unknown(&context));
        let function = module
            .body()
            .append_operation(create_function(&context, "foo"));
        let mut table = SymbolTable::new(module.as_operation()).unwrap();
        let other_function = module
            .body()
            .append_operation(create_function(&context, "foo"));

        assert_eq!(table.insert(other_function).unwrap().value(), Ok("foo_0"));
        assert_eq!(
            other_function.attribute("sym_name"),
            Ok(StringAttribute::new(&context, "foo_0").into())
        );
        assert_eq!(table.lookup("foo"), Some(function));
        assert_eq!(table.lookup("foo_0"), Some(other_function));
        assert!(module.as_operation().verify());
    }

    #[test]
    fn insert_with_name_conflict_and_symbol_use() {
        let context = create_test_context();
        context.set_allow_unregistered_dialects(true);
        let module = Module::parse(&context, "func.func private @foo()").unwrap();
        let mut table = SymbolTable::new(module.as_operation()).unwrap();
        let user = module.body().append_operation(
            OperationBuilder::new("bar", Location::unknown(&context))
                .add_attributes(&[(
                    Identifier::new(&context, "callee"),
                    FlatSymbolRefAttribute::new(&context, "foo").into(),
                )])
                .build()
                .unwrap(),
        );

        assert_eq!(
            table
                .insert(
                    module
                        .body()
                        .append_operation(create_function(&context, "foo"))
                )
                .unwrap()
                .value(),
            Ok("foo_0")
        );
        assert_eq!(
            user.attribute("callee"),
            Ok(FlatSymbolRefAttribute::new(&context, "foo").into())
        );
    }

    #[test]
    fn insert_held_symbol() {
        let context = create_test_context();
        let module = Module::new(Location::unknown(&context));
        let function = module
            .body()
            .append_operation(create_function(&context, "foo"));
        let mut table = SymbolTable::new(module.as_operation()).unwrap();

        assert_eq!(table.insert(function).unwrap().value(), Ok("foo"));
        assert_eq!(
            function.attribute("sym_name"),
            Ok(StringAttribute::new(&context, "foo").into())
        );
        assert_eq!(table.lookup("foo"), Some(function));
        assert_eq!(table.lookup("foo_0"), None);
    }

    #[test]
    fn insert_symbol_renamed_after_creation() {
        let context = create_test_context();
        let module = Module::new(Location::unknown(&context));
        let function = module
            .body()
            .append_operation(create_function(&context, "foo"));
        let mut table = SymbolTable::new(module.as_operation()).unwrap();

        module
            .body()
            .first_operation_mut()
            .unwrap()
            .set_attribute("sym_name", StringAttribute::new(&context, "bar").into());

        assert_eq!(table.insert(function).unwrap().value(), Ok("bar"));
        assert_eq!(table.lookup("foo"), Some(function));
        assert_eq!(table.lookup("bar"), Some(function));
    }

    #[test]
    fn fail_to_insert_non_symbol() {
        let context = create_test_context();
        context.set_allow_unregistered_dialects(true);
        let module = Module::new(Location::unknown(&context));
        let operation = module.body().append_operation(create_operation(&context));

        assert_eq!(
            SymbolTable::new(module.as_operation())
                .unwrap()
                .insert(operation)
                .unwrap_err(),
            Error::SymbolExpected(operation.to_string())
        );
    }

    #[test]
    fn fail_to_insert_symbol_with_non_string_name() {
        let context = create_test_context();
        context.set_allow_unregistered_dialects(true);
        let module = Module::new(Location::unknown(&context));
        let operation = module.body().append_operation(create_operation_with_name(
            &context,
            Attribute::unit(&context),
        ));

        assert_eq!(
            SymbolTable::new(module.as_operation())
                .unwrap()
                .insert(operation)
                .unwrap_err(),
            Error::SymbolExpected(operation.to_string())
        );
    }

    #[test]
    fn fail_to_insert_symbol_with_typed_name() {
        let context = create_test_context();
        context.set_allow_unregistered_dialects(true);
        let module = Module::new(Location::unknown(&context));
        let operation = module.body().append_operation(create_operation_with_name(
            &context,
            StringAttribute::typed(IntegerType::new(&context, 32).into(), "foo").into(),
        ));

        assert_eq!(
            SymbolTable::new(module.as_operation())
                .unwrap()
                .insert(operation)
                .unwrap_err(),
            Error::SymbolExpected(operation.to_string())
        );
    }

    #[test]
    fn fail_to_insert_symbol_in_another_symbol_table() {
        let context = create_test_context();
        let module = Module::new(Location::unknown(&context));
        let mut table = SymbolTable::new(module.as_operation()).unwrap();

        assert_eq!(
            table
                .insert(
                    Module::new(Location::unknown(&context))
                        .body()
                        .append_operation(create_function(&context, "foo"))
                )
                .unwrap_err(),
            Error::SymbolNotFound("foo".into())
        );
        assert_eq!(table.lookup("foo"), None);
    }

    #[test]
    fn fail_to_insert_symbol_in_nested_symbol_table() {
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
        let mut table = SymbolTable::new(module.as_operation()).unwrap();

        assert_eq!(
            table
                .insert(
                    module
                        .body()
                        .first_operation()
                        .unwrap()
                        .first_region()
                        .unwrap()
                        .first_block()
                        .unwrap()
                        .first_operation()
                        .unwrap()
                )
                .unwrap_err(),
            Error::SymbolNotFound("bar".into())
        );
        assert_eq!(table.lookup("bar"), None);
    }

    #[test]
    fn fail_to_insert_detached_symbol() {
        let context = create_test_context();
        let module = Module::new(Location::unknown(&context));
        let mut table = SymbolTable::new(module.as_operation()).unwrap();

        assert_eq!(
            table
                .insert(
                    Module::parse(&context, "module @foo {}")
                        .unwrap()
                        .as_operation()
                )
                .unwrap_err(),
            Error::SymbolNotFound("foo".into())
        );
        assert_eq!(table.lookup("foo"), None);
        assert_eq!(module.body().first_operation(), None);
    }

    #[test]
    fn fail_to_insert_symbol_in_moved_block() {
        let context = create_test_context();
        let module = Module::new(Location::unknown(&context));
        let region = create_region(1);
        let function = module
            .body()
            .append_operation(create_function(&context, "foo"));
        let mut table = SymbolTable::new(module.as_operation()).unwrap();

        IrRewriter::new(&context)
            .as_rewriter_base()
            .move_block_before(module.body(), region.first_block().unwrap());

        assert_eq!(
            table.insert(function).unwrap_err(),
            Error::SymbolNotFound("foo".into())
        );
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
    fn erase_symbol_inserted_after_creation() {
        let context = create_test_context();
        let module = Module::new(Location::unknown(&context));
        let mut table = SymbolTable::new(module.as_operation()).unwrap();
        let function = module
            .body()
            .append_operation(create_function(&context, "foo"));

        table.insert(function).unwrap();

        assert_eq!(unsafe { table.erase(function) }, Ok(()));
        assert_eq!(table.lookup("foo"), None);
        assert_eq!(module.body().first_operation(), None);
    }

    #[test]
    fn fail_to_erase_non_symbol() {
        let context = create_test_context();
        context.set_allow_unregistered_dialects(true);
        let module = Module::new(Location::unknown(&context));
        let operation = module.body().append_operation(create_operation(&context));

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
    fn fail_to_erase_symbol_with_typed_name() {
        let context = create_test_context();
        let module = Module::new(Location::unknown(&context));
        let function = module
            .body()
            .append_operation(create_function(&context, "foo"));
        let mut table = SymbolTable::new(module.as_operation()).unwrap();

        module.body().first_operation_mut().unwrap().set_attribute(
            "sym_name",
            StringAttribute::typed(IntegerType::new(&context, 32).into(), "foo").into(),
        );

        assert_eq!(
            unsafe { table.erase(function) },
            Err(Error::SymbolExpected(function.to_string()))
        );
        assert_eq!(table.lookup("foo"), Some(function));
        assert_eq!(module.body().first_operation(), Some(function));
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
            Err(Error::SymbolNotFound("foo".into()))
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
            Err(Error::SymbolNotFound("foo".into()))
        );
    }

    #[test]
    fn fail_to_erase_symbol_appended_after_creation() {
        let context = create_test_context();
        let module = Module::new(Location::unknown(&context));
        let mut table = SymbolTable::new(module.as_operation()).unwrap();
        let function = module
            .body()
            .append_operation(create_function(&context, "foo"));

        assert_eq!(
            unsafe { table.erase(function) },
            Err(Error::SymbolNotFound("foo".into()))
        );
        assert_eq!(module.body().first_operation(), Some(function));
    }

    #[test]
    fn fail_to_erase_symbol_renamed_after_creation() {
        let context = create_test_context();
        let module = Module::new(Location::unknown(&context));
        let function = module
            .body()
            .append_operation(create_function(&context, "foo"));
        let mut table = SymbolTable::new(module.as_operation()).unwrap();

        module
            .body()
            .first_operation_mut()
            .unwrap()
            .set_attribute("sym_name", StringAttribute::new(&context, "bar").into());

        assert_eq!(
            unsafe { table.erase(function) },
            Err(Error::SymbolNotFound("bar".into()))
        );
        assert_eq!(module.body().first_operation(), Some(function));
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
