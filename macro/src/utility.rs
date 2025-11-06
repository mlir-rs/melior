use regex::{Captures, Regex};
use std::sync::LazyLock;

static NAME_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"(bf_16|f_16|f_32|f_64|i_8|i_16|i_32|i_64|float_8_e_[0-9]_m_[0-9](_fn)?)"#)
        .unwrap()
});

pub fn map_name(name: &str) -> String {
    NAME_PATTERN
        .replace_all(name, |captures: &Captures| {
            captures.get(0).unwrap().as_str().replace('_', "")
        })
        .to_string()
        .trim_end_matches('_')
        .into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_normal_type_name() {
        assert_eq!(map_name("index"), "index");
        assert_eq!(map_name("integer"), "integer");
    }

    #[test]
    fn map_integer_type_name() {
        assert_eq!(map_name("i_64"), "i64");
    }

    #[test]
    fn map_float_type_name() {
        assert_eq!(map_name("f_64"), "f64");
        assert_eq!(map_name("float_8_e_5_m_2"), "float8e5m2");
        assert_eq!(map_name("float_8_e_4_m_3_fn"), "float8e4m3fn");
    }
}
