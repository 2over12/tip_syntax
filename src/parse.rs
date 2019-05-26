use super::ErrorType;
use super::ParseError;
use super::SyntaxTree;
use crate::lex::Token;
use crate::lex::TokenType;
use crate::tree::Expr;
use crate::tree::Function;
use crate::tree::Identifier;
use crate::tree::Literal;
use crate::tree::Stmt;
use crate::tree::{BinaryOp, UnaryOp};
use std::iter::Peekable;
use std::collections::HashMap;

type Result<R> = std::result::Result<R, ParseError>;

struct TokenStream<T: Iterator<Item = Token>> {
    tokens: Peekable<T>,
    /// Last consumed
    previous: Option<Token>,
}

impl<'a, T: Iterator<Item = Token>> TokenStream<T> {
    fn new(tokens: T) -> TokenStream<T> {
        TokenStream {
            tokens: tokens.peekable(),
            previous: None,
        }
    }

    fn is_empty(&mut self) -> bool {
        self.peek().is_none()
    }

    fn get_prev(&self) -> Option<&Token> {
        self.previous.as_ref()
    }

    fn advance(&mut self) -> Option<&Token> {
        self.previous = self.tokens.next();
        self.get_prev()
    }

    fn peek(&mut self) -> Option<&Token> {
        self.tokens.peek()
    }

    fn peek_match(&mut self, ty: &Vec<TokenType>) -> bool {
        ty.into_iter().any(|x| match self.peek() {
            Some(tk) => *x == tk.get_type(),
            None => false,
        })
    }

    fn take_previous(&mut self) -> Option<Token> {
        self.previous.take()
    }

    fn take_match(&mut self, tys: &Vec<TokenType>) -> bool {
        if self.peek_match(tys) {
            self.advance();
            true
        } else {
            false
        }
    }

    fn consume(&mut self, expected_ty: &Vec<TokenType>, msg: &'static str) -> Result<Token> {
        if self.take_match(expected_ty) {
            Ok(self.previous.take().unwrap())
        } else {
            Err(if let Some(tk) = self.get_prev() {
                ParseError::new(ErrorType::ExpectedToken(msg), tk.get_loc())
            } else {
                ParseError::new(ErrorType::ExpectedToken(msg), 0..0)
            })
        }
    }
}

pub(super) fn parse(tks: impl Iterator<Item = Token>) -> super::Result<SyntaxTree> {
    Parser::new(TokenStream::new(tks)).parse()
}

struct Parser<T: Iterator<Item = Token>> {
    tokens: TokenStream<T>,
    errors: Vec<ParseError>,
}

impl<'a, T: Iterator<Item = Token>> Parser<T> {
    fn new(tks: TokenStream<T>) -> Parser<T> {
        Parser {
            tokens: tks,
            errors: Vec::new(),
        }
    }

    fn parse(mut self) -> super::Result<SyntaxTree> {
        let mut functions = Vec::new();
        while !self.tokens.is_empty() && !self.tokens.take_match(&vec![TokenType::Eof]) {
            match self.function() {
                Ok(f) => functions.push(f),
                // TODO Add syncing here
                Err(f) => {
                    self.errors.push(f);
                    return Err(self.errors);
                }
            }
        }

        if self.errors.is_empty() {
            Ok(SyntaxTree::new(functions))
        } else {
            Err(self.errors)
        }
    }

    fn function(&mut self) -> Result<Function> {
        let name = self
            .tokens
            .consume(&vec![TokenType::Identifier], "Expected function name")?;
        self.tokens
            .consume(&vec![TokenType::LeftParen], "Expected '('")?;
        let mut params = Vec::new();

        while !self.tokens.is_empty() && !self.tokens.peek_match(&vec![TokenType::RightParen]) {
            let param_name = self
                .tokens
                .consume(&vec![TokenType::Identifier], "Expected param name")?;
            params.push(Identifier::from(param_name));
            if !self.tokens.peek_match(&vec![TokenType::RightParen]) {
                self.tokens.consume(
                    &vec![TokenType::Comma],
                    "Expected ',' before next parameter.",
                )?;
            }
        }

        self.tokens.consume(
            &vec![TokenType::RightParen],
            "Expected ')' to close function arguments",
        )?;
        self.tokens.consume(
            &vec![TokenType::LeftBracket],
            "Expected '{' to open function body",
        )?;

        let decls = self.declerations()?;
        let blk = self.statement()?;
        let ret = self.parse_return()?;
        self.tokens.consume(
            &vec![TokenType::RightBracket],
            "Expected '}' to close function body",
        )?;
        Ok(Function::new(
            Identifier::from(name),
            params,
            decls,
            blk,
            ret,
        ))
    }

    fn parse_return(&mut self) -> Result<Expr> {
        self.tokens.consume(
            &vec![TokenType::Return],
            "Expected return at end of function.",
        )?;
        let exp = self.expression()?;
        self.tokens
            .consume(&vec![TokenType::Semi], "Expected ';' after return.")?;
        Ok(exp)
    }
    fn declerations(&mut self) -> Result<Vec<Identifier>> {
        let mut assigns = Vec::new();
        if self.tokens.take_match(&vec![TokenType::Var]) {
            let name = self
                .tokens
                .consume(&vec![TokenType::Identifier], "Expected name of var.")?;
            assigns.push(Identifier::from(name));
            while self.tokens.take_match(&vec![TokenType::Comma]) {
                let name = self
                    .tokens
                    .consume(&vec![TokenType::Identifier], "Expected name of var.")?;
                assigns.push(Identifier::from(name));
            }

            self.tokens.consume(
                &vec![TokenType::Semi],
                "Expected ';' after var declerations",
            )?;
        }
        Ok(assigns)
    }

    // TODO add syncing here
    fn statement(&mut self) -> Result<Stmt> {
        let first = self.atomic_statement()?;

        Ok(
            if self.tokens.peek_match(&vec![
                TokenType::Identifier,
                TokenType::Output,
                TokenType::While,
                TokenType::If,
                TokenType::Star,
            ]) {
                let following = Box::new(self.statement()?);
                Stmt::Join(Box::new(first), following)
            } else {
                first
            },
        )
    }

    fn expression(&mut self) -> Result<Expr> {
        self.comparison()
    }

    fn comparison(&mut self) -> Result<Expr> {
        self.binary_left_assoc(
            |x| x.linear_arith(),
            vec![TokenType::EqualEqual, TokenType::Greater],
        )
    }

    fn mul_div(&mut self) -> Result<Expr> {
        self.binary_left_assoc(
            |x| x.dot_operator(),
            vec![TokenType::Star, TokenType::Slash],
        )
    }

    fn linear_arith(&mut self) -> Result<Expr> {
        self.binary_left_assoc(|x| x.mul_div(), vec![TokenType::Sub, TokenType::Plus])
    }

    fn dot_operator(&mut self) -> Result<Expr> {
    	 self.general_op_seq(
            |x| x.unary_operator(),
            |x| x.tokens.consume(&vec![TokenType::Identifier], "Expected identifier after '.'"),
            |left,_,right| {Expr::Projection(Box::new(left), Identifier::from(right))},
            vec![TokenType::Dot],
        )
    }

    fn unary_operator(&mut self) -> Result<Expr> {
        if self.tokens.take_match(&vec![
            TokenType::Sub,
            TokenType::Alloc,
            TokenType::Star,
            TokenType::Ref,
        ]) {
            let op = self.tokens.take_previous().unwrap();
            let right = self.unary_operator()?;
            Ok(Expr::UnOp(UnaryOp::from(op), Box::new(right)))
        } else {
            self.call()
        }
    }

    fn call(&mut self) -> Result<Expr> {
        let mut left = self.primary()?;

        while self.tokens.take_match(&vec![TokenType::LeftParen]) {
            left = self.finish_call(left)?;
        }

        Ok(left)
    }

    fn finish_record(&mut self) -> Result<Expr> {
    	let mut rec_map = HashMap::new();

    	if !self.tokens.peek_match(&vec![TokenType::RightBracket]) {
    		loop {
    			let name = Identifier::from(self.tokens.consume(&vec![TokenType::Identifier], "Expected identifier in record")?);
    			self.tokens.consume(&vec![TokenType::Colon],"Expected ':' after identifier in record")?;
    			let exp = self.expression()?;
    			rec_map.insert(name,exp);

    			if !self.tokens.take_match(&vec![TokenType::Comma]) {
    				break;
    			}
    		}
    	}

    	self.tokens.consume(&vec![TokenType::RightBracket], "Expected '}' to close record")?;
    	Ok(Expr::Record(rec_map))
    }

    fn finish_call(&mut self, func: Expr) -> Result<Expr> {
        let mut args = Vec::new();

        if !self.tokens.peek_match(&vec![TokenType::RightParen]) {
            loop {
                args.push(self.expression()?);

                if !self.tokens.take_match(&vec![TokenType::Comma]) {
                    break;
                }
            }
        }

        self.tokens.consume(
            &vec![TokenType::RightParen],
            "expected arguments to be closed by ')'",
        )?;
        Ok(Expr::App(Box::new(func), args))
    }

    fn primary(&mut self) -> Result<Expr> {
        if self.is_value() {
            self.value()
        } else if self.tokens.take_match(&vec![TokenType::LeftParen]) {
            let grp = Box::new(self.expression()?);
            self.tokens
                .consume(&vec![TokenType::RightParen], "Expected ')' after group")?;
            Ok(Expr::Grouping(grp))
        } else if self.tokens.take_match(&vec![TokenType::Input]) {
            Ok(Expr::Input)
        } else if self.tokens.take_match(&vec![TokenType::LeftBracket]) {
        	self.finish_record()
        } else {
            Err(ParseError::new(
                ErrorType::ExpectedToken("unexpected token"),
                0..0,
            ))
        }
    }

    fn is_value(&mut self) -> bool {
        self.tokens.peek_match(&vec![
            TokenType::Identifier,
            TokenType::Int,
            TokenType::Null,
        ])
    }

    fn value(&mut self) -> Result<Expr> {
        self.tokens.advance();
        Ok(Expr::Value(Literal::from(
            self.tokens.take_previous().unwrap(),
        )))
    }

    fn binary_left_assoc(
        &mut self,
        higher_prec: impl Fn(&mut Parser<T>) -> Result<Expr>,
        matchees: Vec<TokenType>,
    ) -> Result<Expr> {
        self.list_operand(
            higher_prec,
            |left, op, right| Expr::BinOp(BinaryOp::from(op), Box::new(left), Box::new(right)),
            matchees,
        )
    }

    fn list_operand(
        &mut self,
        higher_prec: impl Fn(&mut Parser<T>) -> Result<Expr>,
        joiner: impl Fn(Expr, Token, Expr) -> Expr,
        matching: Vec<TokenType>,
    ) -> Result<Expr> {
    	self.general_op_seq(&higher_prec, &
    		higher_prec, joiner, matching)
    }

       fn general_op_seq<U>(
        &mut self,
        higher_prec: impl Fn(&mut Parser<T>) -> Result<Expr>,
        next_val: impl Fn(&mut Parser<T>) -> Result<U>,
        joiner: impl Fn(Expr, Token, U) -> Expr,
        matching: Vec<TokenType>,
    ) -> Result<Expr> {
        let mut lhand = higher_prec(self)?;
        while self.tokens.take_match(&matching) {
            let op = self.tokens.take_previous().unwrap();
            let right = next_val(self)?;
            lhand = joiner(lhand, op, right);
        }

        Ok(lhand)
    }

    fn atomic_statement(&mut self) -> Result<Stmt> {
        if self.tokens.take_match(&vec![TokenType::While]) {
            self.tokens
                .consume(&vec![TokenType::LeftParen], "Expected '(' after while")?;
            let cond = self.expression()?;
            self.tokens
                .consume(&vec![TokenType::RightParen], "Expected ')' after while")?;
            self.tokens
                .consume(&vec![TokenType::LeftBracket], "Expected '{' open body")?;
            let stmt = Box::new(self.statement()?);
            self.tokens
                .consume(&vec![TokenType::RightBracket], "Expected '}' close body")?;

            Ok(Stmt::While(cond, stmt))
        } else if self.tokens.take_match(&vec![TokenType::Output]) {
            let value = self.expression()?;
            self.tokens
                .consume(&vec![TokenType::Semi], "Expected ';' after output")?;
            Ok(Stmt::Output(value))
        } else if self.tokens.take_match(&vec![TokenType::If]) {
            self.tokens.consume(
                &vec![TokenType::LeftParen],
                "Expected '(' before condition in if",
            )?;
            let cond = self.expression()?;

            self.tokens.consume(
                &vec![TokenType::RightParen],
                "Expected ')' after condition in if",
            )?;
            self.tokens.consume(
                &vec![TokenType::LeftBracket],
                "Expected '{' to open body of if",
            )?;
            let if_body = self.statement()?;
            self.tokens.consume(
                &vec![TokenType::RightBracket],
                "Expected '}' to close body of if",
            )?;

            let else_body = if self.tokens.take_match(&vec![TokenType::Else]) {
                self.tokens.consume(
                    &vec![TokenType::LeftBracket],
                    "Expected '{' to open body of else",
                )?;
                let ebd = self.statement()?;
                self.tokens.consume(
                    &vec![TokenType::RightBracket],
                    "Expected '}' to close body of else",
                )?;
                Some(ebd)
            } else {
                None
            };

            Ok(Stmt::If(cond, Box::new(if_body), Box::new(else_body)))
        } else if self.tokens.take_match(&vec![TokenType::Identifier]) {
            let ident = Identifier::from(self.tokens.take_previous().unwrap());
            self.tokens
                .consume(&vec![TokenType::Equal], "Expected equal after assignment")?;
            let val = self.expression()?;
            self.tokens
                .consume(&vec![TokenType::Semi], "Expected ';' after assignment")?;
            Ok(Stmt::Assignment(ident, val))
        } else if self.tokens.take_match(&vec![TokenType::Star]) {
            let ident = Identifier::from(self.tokens.consume(
                &vec![TokenType::Identifier],
                "expected ident for derefed assignment",
            )?);
            self.tokens
                .consume(&vec![TokenType::Equal], "Expected equal after assignment")?;
            let val = self.expression()?;
            self.tokens
                .consume(&vec![TokenType::Semi], "Expected ';' after assignment")?;
            Ok(Stmt::DerefAssignment(ident, val))
        } else {
            self.tokens.advance();
            Err(ParseError::new(
                ErrorType::ExpectedToken("expected statement"),
                0..0,
            ))
        }
    }
}
