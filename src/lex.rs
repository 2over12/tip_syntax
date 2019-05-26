use crate::Result;
use logos::Logos;

use crate::ErrorType;
use crate::ParseError;
use core::ops::Range;

#[derive(Debug, Logos, PartialEq, Clone)]
pub(crate) enum TokenType {
    #[end]
    Eof,

    #[error]
    Error,

    // Atoms
    #[regex = "[a-zA-Z][a-zA-Z0-9]*"]
    Identifier,
    #[regex = "[0-9]+"]
    Int,

    // Keywords
    #[token = "if"]
    If,
    #[token = "else"]
    Else,
    #[token = "while"]
    While,
    #[token = "output"]
    Output,
    #[token = "input"]
    Input,
    #[token = "var"]
    Var,
    #[token = "null"]
    Null,
    #[token = "alloc"]
    Alloc,
    #[token = "return"]
    Return,

    // Operators
    #[token = "+"]
    Plus,
    #[token = "-"]
    Sub,
    #[token = "*"]
    Star,
    #[token = "/"]
    Slash,
    #[token = ">"]
    Greater,
    #[token = "=="]
    EqualEqual,
    #[token = "="]
    Equal,
    #[token = "&"]
    Ref,

    // Markers
    #[token = "("]
    LeftParen,
    #[token = ")"]
    RightParen,
    #[token = "{"]
    LeftBracket,
    #[token = "}"]
    RightBracket,
    #[token = ","]
    Comma,
    #[token = ";"]
    Semi,
    #[token = "."]
    Dot,
    #[token = ":"]
    Colon,
}

#[derive(Debug)]
pub(crate) struct Token {
    loc: Range<usize>,
    ty: TokenType,
    lexeme: String,
}

impl Token {
    fn new(ty: TokenType, loc: Range<usize>, lexeme: String) -> Token {
        Token { ty, loc, lexeme }
    }

    pub fn get_type(&self) -> TokenType {
        self.ty.clone()
    }

    pub fn get_lexeme(&self) -> &str {
        &self.lexeme
    }

    pub fn get_loc(&self) -> Range<usize> {
        self.loc.clone()
    }

    pub fn to_lexeme(self) -> String {
        self.lexeme
    }
}

pub(super) fn lex(prog: &str) -> Result<Vec<Token>> {
    println!("Starting lexing");
    let mut lexer = TokenType::lexer(prog);

    let mut tokens = Vec::new();
    let mut errors = Vec::new();

    loop {
        match &lexer.token {
            TokenType::Error => {
                errors.push(ParseError::new(ErrorType::UnexpectedToken, lexer.range()));
            }
            TokenType::Eof => break,
            x => tokens.push(Token::new(
                x.clone(),
                lexer.range(),
                lexer.slice().to_owned(),
            )),
        }
        lexer.advance();
    }

    if errors.len() > 0 {
        Err(errors)
    } else {
        Ok(tokens)
    }
}
