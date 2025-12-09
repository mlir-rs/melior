use super::error::Error;
use comrak::{arena_tree::NodeEdge, format_commonmark, nodes::NodeValue, parse_document, Arena};
use convert_case::{Case, Casing};
use proc_macro2::Ident;
use quote::format_ident;
use syn::{parse_quote, Type};

const RESERVED_NAMES: &[&str] = &["name", "operation", "builder"];

pub fn generate_result_type(r#type: Type) -> Type {
    parse_quote!(Result<#r#type, ::melior::Error>)
}

pub fn generate_iterator_type(r#type: Type) -> Type {
    parse_quote!(impl Iterator<Item = #r#type>)
}

pub fn sanitize_snake_case_identifier(name: &str) -> Result<Ident, Error> {
    sanitize_name(&name.to_case(Case::Snake))
}

fn sanitize_name(name: &str) -> Result<Ident, Error> {
    // Replace any "." with "_".
    let mut name = name.replace('.', "_").trim_end_matches('_').to_string();

    // Add "_" suffix to avoid conflicts with existing methods.
    if RESERVED_NAMES.contains(&name.as_str())
        || name
            .chars()
            .next()
            .ok_or_else(|| Error::InvalidIdentifier(name.clone()))?
            .is_numeric()
    {
        name = format!("_{name}");
    }

    // Try to parse the string as an ident, and prefix the identifier
    // with "r#" if it is not a valid identifier.
    Ok(syn::parse_str::<Ident>(&name).unwrap_or_else(|_| format_ident!("r#{}", name)))
}

pub fn sanitize_documentation(string: &str) -> Result<String, Error> {
    let arena = Arena::new();
    let node = parse_document(&arena, &unindent::unindent(string), &Default::default());

    for node in node.traverse() {
        let NodeEdge::Start(node) = node else {
            continue;
        };
        let mut ast = node.data.borrow_mut();
        let NodeValue::CodeBlock(block) = &mut ast.value else {
            continue;
        };

        if block.info.is_empty() {
            // Mark them not in Rust to prevent documentation tests.
            block.info = "text".into();
        }
    }

    let mut buffer = String::with_capacity(string.len());

    format_commonmark(node, &Default::default(), &mut buffer)?;

    Ok(buffer)
}

pub fn capitalize_string(string: &str) -> String {
    if string.is_empty() {
        "".into()
    } else {
        string[..1].to_uppercase() + &string[1..]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    #[test]
    fn sanitize_name_with_dot() {
        assert_eq!(
            sanitize_snake_case_identifier("foo.bar").unwrap(),
            "foo_bar"
        );
    }

    #[test]
    fn sanitize_name_with_dot_and_underscore() {
        assert_eq!(
            sanitize_snake_case_identifier("foo.bar_baz").unwrap(),
            "foo_bar_baz"
        );
    }

    #[test]
    fn sanitize_reserved_name() {
        assert_eq!(
            sanitize_snake_case_identifier("builder").unwrap(),
            "_builder"
        );
    }

    #[test]
    fn sanitize_code_block() {
        assert_eq!(
            &sanitize_documentation("```\nfoo\n```\n").unwrap(),
            "```text\nfoo\n```\n"
        );
    }

    #[test]
    fn sanitize_code_blocks() {
        assert_eq!(
            &sanitize_documentation("```\nfoo\n```\n\n```\nbar\n```\n").unwrap(),
            "```text\nfoo\n```\n\n``` text\nbar\n```\n"
        );
    }

    #[test]
    fn capitalize() {
        assert_eq!(&capitalize_string("foo"), "Foo");
    }
}
