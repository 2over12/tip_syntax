use std::collections::HashMap;
use crate::lex::Token;
use crate::lex::TokenType;
#[derive(PartialEq, Debug)]
pub enum Stmt {
    Output(Expr),
    If(Expr, Box<Stmt>, Box<Option<Stmt>>),
    While(Expr, Box<Stmt>),
    Join(Box<Stmt>, Box<Stmt>),
    Assignment(Identifier, Expr),
    DerefAssignment(Identifier, Expr),
}

impl Stmt {
    pub fn visit_stmt<R>(&self, visitor: impl StmtVisitor<R>) -> R {
        match self {
            Stmt::Output(exp) => visitor.visit_output(exp),
            Stmt::If(disc, body, otherwise) => visitor.visit_if(disc, body, otherwise),
            Stmt::While(disc, body) => visitor.visit_while(disc, body),
            Stmt::Join(first, second) => visitor.visit_join(first, second),
            Stmt::Assignment(name, exp) => visitor.visit_assignment(name, exp),
            Stmt::DerefAssignment(name, exp) => visitor.visit_deref_assignment(name, exp),
        }
    }
}

pub trait StmtVisitor<R> {
    fn visit_assignment(self, name: &Identifier, expr: &Expr) -> R;
    fn visit_deref_assignment(self, name: &Identifier, expr: &Expr) -> R;
    fn visit_output(self, value: &Expr) -> R;
    fn visit_if(self, discriminant: &Expr, body: &Stmt, otherwise: &Option<Stmt>) -> R;
    fn visit_while(self, discriminant: &Expr, body: &Stmt) -> R;
    fn visit_join(self, first: &Stmt, second: &Stmt) -> R;
}

#[derive(PartialEq, Debug, Clone)]
pub enum Expr {
    Value(Literal),
    Grouping(Box<Expr>),
    BinOp(BinaryOp, Box<Expr>, Box<Expr>),
    UnOp(UnaryOp, Box<Expr>),
    App(Box<Expr>, Vec<Expr>),
    Record(HashMap<Identifier, Expr>),
    Projection(Box<Expr>, Identifier),
    Input,
}

pub trait ExprVisitor<R> {
    fn visit_value(self, literal: &Literal) -> R;
    fn visit_grouping(self, inner: &Expr) -> R;
    fn visit_bin_op(self, op: &BinaryOp, left: &Expr, right: &Expr) -> R;
    fn visit_un_op(self, op: &UnaryOp, rand: &Expr);
    fn visit_application(self, applyee: &Expr, args: &[Expr]) -> R;
    fn visit_record(self, bindings: &[(Identifier, Expr)]) -> R;
    fn visit_projection(self, record: &Expr, field: Identifier) -> R;
    fn visit_input(self) -> R;
}

#[derive(Debug, PartialEq, Clone)]
pub enum BinaryOp {
    Add,
    Sub,
    Mult,
    Div,
    Greater,
    Equal,
}

impl From<Token> for BinaryOp {
    fn from(tk: Token) -> Self {
        match tk.get_type() {
            TokenType::Greater => BinaryOp::Greater,
            TokenType::Plus => BinaryOp::Add,
            TokenType::Sub => BinaryOp::Sub,
            TokenType::Star => BinaryOp::Mult,
            TokenType::Slash => BinaryOp::Div,
            TokenType::EqualEqual => BinaryOp::Equal,
            _ => panic!("Should not be possible to reach binop without binop"),
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum UnaryOp {
    Ref,
    Deref,
    Alloc,
    Neg,
}

impl From<Token> for UnaryOp {
    fn from(tk: Token) -> Self {
        match tk.get_type() {
            TokenType::Alloc => UnaryOp::Alloc,
            TokenType::Star => UnaryOp::Deref,
            TokenType::Ref => UnaryOp::Ref,
            TokenType::Sub => UnaryOp::Neg,
            _ => unreachable!(),
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum Literal {
    Number(i64),
    Identifier(Identifier),
    Null,
}

impl From<Token> for Literal {
    fn from(tk: Token) -> Self {
        match tk.get_type() {
            TokenType::Identifier => Literal::Identifier(Identifier(tk.to_lexeme())),
            TokenType::Int => Literal::Number(tk.to_lexeme().parse().unwrap()),
            TokenType::Null => Literal::Null,
            _ => panic!("Should not be possible to reach binop without binop"),
        }
    }
}

#[derive(Debug, PartialEq)]
pub struct SyntaxTree {
    functions: Vec<Function>,
}

impl SyntaxTree {
    pub(crate) fn new(functions: Vec<Function>) -> SyntaxTree {
        SyntaxTree { functions }
    }
}

#[derive(Debug, PartialEq, Clone, Eq, Hash)]
pub struct Identifier(String);
impl Identifier {
    #[cfg(test)]
    pub(crate) fn new(name: String) -> Identifier {
        Identifier(name)
    }
}

impl From<Token> for Identifier {
    fn from(val: Token) -> Identifier {
        Identifier(val.to_lexeme())
    }
}

#[derive(Debug, PartialEq)]
pub struct Function(Identifier, Vec<Identifier>, Vec<Identifier>, Stmt, Expr);

impl Function {
    pub fn new(
        name: Identifier,
        params: Vec<Identifier>,
        vars: Vec<Identifier>,
        body: Stmt,
        ret_val: Expr,
    ) -> Function {
        Function(name, params, vars, body, ret_val)
    }
}
