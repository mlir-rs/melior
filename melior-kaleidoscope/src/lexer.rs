//! A lexer of the Toy language.

use crate::{error::Error, location::Location};
use core::{
    fmt::{self, Display, Formatter},
    str::Chars,
};

/// A token.
#[derive(Clone, Debug, PartialEq)]
pub enum Token {
    /// A `def` keyword.
    Def,
    /// A `return` keyword.
    Return,
    /// A `struct` keyword.
    Struct,
    /// A `var` keyword.
    Var,
    /// An identifier.
    Identifier(String),
    /// A number.
    Number(f64),
    /// A `,` character.
    Comma,
    /// A `.` character.
    Dot,
    /// A `=` character.
    Equal,
    /// A `-` character.
    Minus,
    /// A `+` character.
    Plus,
    /// A `;` character.
    Semicolon,
    /// A `*` character.
    Star,
    /// A `<` character.
    AngleBracketOpen,
    /// A `>` character.
    AngleBracketClose,
    /// A `{` character.
    BraceOpen,
    /// A `}` character.
    BraceClose,
    /// A `[` character.
    BracketOpen,
    /// A `]` character.
    BracketClose,
    /// A `(` character.
    ParenthesisOpen,
    /// A `)` character.
    ParenthesisClose,
    /// An end of a file.
    End,
}

impl Display for Token {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        match self {
            Self::Def => write!(formatter, "'def'"),
            Self::Return => write!(formatter, "'return'"),
            Self::Struct => write!(formatter, "'struct'"),
            Self::Var => write!(formatter, "'var'"),
            Self::Identifier(name) => write!(formatter, "identifier '{name}'"),
            Self::Number(number) => write!(formatter, "number {number}"),
            Self::Comma => write!(formatter, "','"),
            Self::Dot => write!(formatter, "'.'"),
            Self::Equal => write!(formatter, "'='"),
            Self::Minus => write!(formatter, "'-'"),
            Self::Plus => write!(formatter, "'+'"),
            Self::Semicolon => write!(formatter, "';'"),
            Self::Star => write!(formatter, "'*'"),
            Self::AngleBracketOpen => write!(formatter, "'<'"),
            Self::AngleBracketClose => write!(formatter, "'>'"),
            Self::BraceOpen => write!(formatter, "'{{'"),
            Self::BraceClose => write!(formatter, "'}}'"),
            Self::BracketOpen => write!(formatter, "'['"),
            Self::BracketClose => write!(formatter, "']'"),
            Self::ParenthesisOpen => write!(formatter, "'('"),
            Self::ParenthesisClose => write!(formatter, "')'"),
            Self::End => write!(formatter, "end of file"),
        }
    }
}

/// A lexer.
pub struct Lexer<'a> {
    characters: Chars<'a>,
    character: Option<char>,
    location: Location,
}

impl<'a> Lexer<'a> {
    /// Creates a lexer.
    pub fn new(source: &'a str) -> Self {
        let mut characters = source.chars();

        Self {
            character: characters.next(),
            characters,
            location: Location::new(1, 1),
        }
    }

    /// Reads a next token and its location.
    pub fn next_token(&mut self) -> Result<(Token, Location), Error> {
        loop {
            match self.character {
                Some(character) if character.is_whitespace() => self.advance(),
                Some('#') => {
                    while !matches!(self.character, None | Some('\n')) {
                        self.advance();
                    }
                }
                _ => break,
            }
        }

        let location = self.location;
        let Some(character) = self.character else {
            return Ok((Token::End, location));
        };

        let token = if character.is_ascii_alphabetic() {
            match self
                .read_while(|character| character.is_ascii_alphanumeric() || character == '_')
                .as_str()
            {
                "def" => Token::Def,
                "return" => Token::Return,
                "struct" => Token::Struct,
                "var" => Token::Var,
                identifier => Token::Identifier(identifier.into()),
            }
        } else if character.is_ascii_digit() {
            let number =
                self.read_while(|character| character.is_ascii_digit() || character == '.');

            Token::Number(number.parse().map_err(|_| Error::Syntax {
                location,
                message: format!("invalid number '{number}'"),
            })?)
        } else {
            self.advance();

            match character {
                ',' => Token::Comma,
                '.' => Token::Dot,
                '=' => Token::Equal,
                '-' => Token::Minus,
                '+' => Token::Plus,
                ';' => Token::Semicolon,
                '*' => Token::Star,
                '<' => Token::AngleBracketOpen,
                '>' => Token::AngleBracketClose,
                '{' => Token::BraceOpen,
                '}' => Token::BraceClose,
                '[' => Token::BracketOpen,
                ']' => Token::BracketClose,
                '(' => Token::ParenthesisOpen,
                ')' => Token::ParenthesisClose,
                _ => {
                    return Err(Error::Syntax {
                        location,
                        message: format!("unexpected character '{character}'"),
                    });
                }
            }
        };

        Ok((token, location))
    }

    fn read_while(&mut self, predicate: impl Fn(char) -> bool) -> String {
        let mut string = String::new();

        while let Some(character) = self.character.filter(|&character| predicate(character)) {
            string.push(character);
            self.advance();
        }

        string
    }

    fn advance(&mut self) {
        self.location = if self.character == Some('\n') {
            Location::new(self.location.line() + 1, 1)
        } else {
            Location::new(self.location.line(), self.location.column() + 1)
        };
        self.character = self.characters.next();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    fn tokenize(source: &str) -> Vec<(Token, Location)> {
        let mut lexer = Lexer::new(source);
        let mut tokens = vec![];

        loop {
            let (token, location) = lexer.next_token().unwrap();
            let end = token == Token::End;

            tokens.push((token, location));

            if end {
                return tokens;
            }
        }
    }

    #[test]
    fn tokenize_empty_source() {
        assert_eq!(tokenize(""), vec![(Token::End, Location::new(1, 1))]);
    }

    #[test]
    fn tokenize_keywords() {
        assert_eq!(
            tokenize("def return struct var"),
            vec![
                (Token::Def, Location::new(1, 1)),
                (Token::Return, Location::new(1, 5)),
                (Token::Struct, Location::new(1, 12)),
                (Token::Var, Location::new(1, 19)),
                (Token::End, Location::new(1, 22)),
            ]
        );
    }

    #[test]
    fn tokenize_identifier() {
        assert_eq!(
            tokenize("foo_bar1"),
            vec![
                (Token::Identifier("foo_bar1".into()), Location::new(1, 1)),
                (Token::End, Location::new(1, 9)),
            ]
        );
    }

    #[test]
    fn tokenize_numbers() {
        assert_eq!(
            tokenize("42 1.5"),
            vec![
                (Token::Number(42.0), Location::new(1, 1)),
                (Token::Number(1.5), Location::new(1, 4)),
                (Token::End, Location::new(1, 7)),
            ]
        );
    }

    #[test]
    fn tokenize_punctuation() {
        assert_eq!(
            tokenize(",.=-+;*<>{}[]()")
                .into_iter()
                .map(|(token, _)| token)
                .collect::<Vec<_>>(),
            vec![
                Token::Comma,
                Token::Dot,
                Token::Equal,
                Token::Minus,
                Token::Plus,
                Token::Semicolon,
                Token::Star,
                Token::AngleBracketOpen,
                Token::AngleBracketClose,
                Token::BraceOpen,
                Token::BraceClose,
                Token::BracketOpen,
                Token::BracketClose,
                Token::ParenthesisOpen,
                Token::ParenthesisClose,
                Token::End,
            ]
        );
    }

    #[test]
    fn skip_comments_and_track_lines() {
        assert_eq!(
            tokenize("# comment\n  var # another\n\tx"),
            vec![
                (Token::Var, Location::new(2, 3)),
                (Token::Identifier("x".into()), Location::new(3, 2)),
                (Token::End, Location::new(3, 3)),
            ]
        );
    }

    #[test]
    fn fail_to_tokenize_invalid_number() {
        assert!(matches!(
            Lexer::new("1.2.3").next_token(),
            Err(Error::Syntax { location, message })
                if location == Location::new(1, 1) && message == "invalid number '1.2.3'"
        ));
    }

    #[test]
    fn fail_to_tokenize_unexpected_character() {
        assert!(matches!(
            Lexer::new("  !").next_token(),
            Err(Error::Syntax { location, message })
                if location == Location::new(1, 3) && message == "unexpected character '!'"
        ));
    }
}
