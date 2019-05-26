mod lex;
mod parse;
mod tree;

use core::ops::Range;

pub use tree::SyntaxTree;

#[derive(Debug)]
pub struct ParseError {
    loc: Range<usize>,
    ty: ErrorType,
}

impl ParseError {
    fn new(ty: ErrorType, loc: Range<usize>) -> ParseError {
        ParseError { ty, loc }
    }
}

#[derive(Debug)]
pub enum ErrorType {
    UnexpectedToken,
    ExpectedToken(&'static str),
}

pub type Result<R> = std::result::Result<R, Vec<ParseError>>;

pub fn parse(contents: &str) -> Result<SyntaxTree> {
    let tokens = lex::lex(contents)?;
    parse::parse(tokens.into_iter())
}

#[cfg(test)]
mod tests {
    use super::parse;
    use super::SyntaxTree;
    use crate::tree::{BinaryOp, Expr, Function, Identifier, Literal, Stmt, UnaryOp};
    use std::collections::HashMap;
    #[test]
    fn iterate_1() {
        let prog = "iterate(n) {
	var f;
	f = 1;
	while (n>0) {
		f = f*n;
		n = n - 1;
	}
	return f;
}
"
        .to_owned();
        let function_id = Identifier::new("iterate".to_owned());
        let n_ident = Identifier::new("n".to_owned());
        let n_expr = Expr::Value(Literal::Identifier(n_ident.clone()));
        let f_ident = Identifier::new("f".to_owned());
        let f_expr = Expr::Value(Literal::Identifier(f_ident.clone()));
        let while_cond = Expr::BinOp(
            BinaryOp::Greater,
            Box::new(n_expr.clone()),
            Box::new(Expr::Value(Literal::Number(0))),
        );
        let fbyn = Expr::BinOp(
            BinaryOp::Mult,
            Box::new(f_expr.clone()),
            Box::new(n_expr.clone()),
        );
        let nminus1 = Expr::BinOp(
            BinaryOp::Sub,
            Box::new(n_expr.clone()),
            Box::new(Expr::Value(Literal::Number(1))),
        );
        let while_stmt = Stmt::While(
            while_cond,
            Box::new(Stmt::Join(
                Box::new(Stmt::Assignment(f_ident.clone(), fbyn)),
                Box::new(Stmt::Assignment(n_ident.clone(), nminus1)),
            )),
        );
        let f_assign = Stmt::Assignment(f_ident.clone(), Expr::Value(Literal::Number(1)));
        let func_iter = Function::new(
            function_id,
            vec![n_ident.clone()],
            vec![f_ident.clone()],
            Stmt::Join(Box::new(f_assign), Box::new(while_stmt)),
            f_expr,
        );
        let parsed = SyntaxTree::new(vec![func_iter]);

        assert_eq!(parse(&prog).unwrap(), parsed);
    }

    #[test]
    fn recurse() {
        let prog = "recurse(n) {
	var f;
	if (n==0) { f=1; }
	else { f=n*recurse(n-1); }
	return f;
}
"
        .to_owned();
        let if_cond = Expr::BinOp(
            BinaryOp::Equal,
            Box::new(Expr::Value(Literal::Identifier(Identifier::new(
                "n".to_string(),
            )))),
            Box::new(Expr::Value(Literal::Number(0))),
        );
        let if_body = Stmt::Assignment(
            Identifier::new("f".to_owned()),
            Expr::Value(Literal::Number(1)),
        );
        let args = vec![Expr::BinOp(
            BinaryOp::Sub,
            Box::new(Expr::Value(Literal::Identifier(Identifier::new(
                "n".to_owned(),
            )))),
            Box::new(Expr::Value(Literal::Number(1))),
        )];
        let recur_call = Expr::App(
            Box::new(Expr::Value(Literal::Identifier(Identifier::new(
                "recurse".to_owned(),
            )))),
            args,
        );
        let mult_res = Expr::BinOp(
            BinaryOp::Mult,
            Box::new(Expr::Value(Literal::Identifier(Identifier::new(
                "n".to_owned(),
            )))),
            Box::new(recur_call),
        );
        let recur_assign = Stmt::Assignment(Identifier::new("f".to_owned()), mult_res);

        assert_eq!(
            parse(&prog).unwrap(),
            SyntaxTree::new(vec![Function::new(
                Identifier::new("recurse".to_string()),
                vec![Identifier::new("n".to_string())],
                vec![Identifier::new("f".to_string())],
                Stmt::If(if_cond, Box::new(if_body), Box::new(Some(recur_assign))),
                Expr::Value(Literal::Identifier(Identifier::new("f".to_owned())))
            )])
        );
    }

    #[test]
    fn complicated() {
        let prog = "foo(p,x) {
	var f,q;
	if (*p==0) { f=1; }
	else { 
		q = alloc 0;
		*q = (*p)-1;
		f=(*p)*(x(q,x)); 
	}
	return f;
}

main() {
	var n;
	n = input;
	return foo(&n,foo);
}
"
        .to_owned();

        let p_ident = Identifier::new("p".to_owned());
        let x_ident = Identifier::new("x".to_owned());

        let parameter = vec![p_ident.clone(),x_ident.clone()];

        let foo_ident = Identifier::new("foo".to_owned());

        let f_ident = Identifier::new("f".to_owned());
        let q_ident = Identifier::new("q".to_owned());
        let q_expr = Expr::Value(Literal::Identifier(q_ident.clone()));
        let x_expr = Expr::Value(Literal::Identifier(x_ident.clone()));

        let vars_foo = vec![f_ident.clone(), q_ident.clone()];
        let p_expr = Expr::Value(Literal::Identifier(p_ident));
        let p_deref = Expr::UnOp(UnaryOp::Deref, Box::new(p_expr.clone()));

        let zero_expr = Expr::Value(Literal::Number(0));
        let one_expr = Expr::Value(Literal::Number(1));
        let if_cond = Expr::BinOp(BinaryOp::Equal, Box::new(p_deref.clone()), Box::new(zero_expr.clone()));

        let f_expr = Expr::Value(Literal::Identifier(f_ident.clone()));
        let f_assign = Stmt::Assignment(f_ident.clone(), one_expr.clone());

        let q_assign = Stmt::Assignment(q_ident.clone(),Expr::UnOp(UnaryOp::Alloc, Box::new(zero_expr.clone())));
        let p_group = Expr::Grouping(Box::new(p_deref.clone()));
        let p_sub = Expr::BinOp(BinaryOp::Sub, Box::new(p_group), Box::new(one_expr.clone()));

        let q_deref = Stmt::DerefAssignment(q_ident, p_sub);
        let x_call = Expr::App(Box::new(x_expr.clone()), vec![q_expr.clone(),x_expr.clone()]);
        let group_call = Expr::Grouping(Box::new(x_call));
        let mult_call = Expr::BinOp(BinaryOp::Mult, Box::new(Expr::Grouping(Box::new(p_deref))),Box::new(group_call));
        let f_else_assign = Stmt::Assignment(f_ident, mult_call);
        let else_body = Stmt::Join(Box::new(q_assign), Box::new(Stmt::Join(Box::new(q_deref), Box::new(f_else_assign))));

        let if_stat = Stmt::If(if_cond,Box::new(f_assign),Box::new(Some(else_body)));

        let foo_func = Function::new(foo_ident.clone(), parameter,vars_foo, if_stat, f_expr);

        let n_ident = Identifier::new("n".to_owned());
        let n_expr = Expr::Value(Literal::Identifier(n_ident.clone()));
        let main_ident = Identifier::new("main".to_owned());
        let n_assign = Stmt::Assignment(n_ident.clone(), Expr::Input);

        let n_ref = Expr::UnOp(UnaryOp::Ref, Box::new(n_expr));
        let foo_expr = Expr::Value(Literal::Identifier(foo_ident.clone()));
        let foo_call = Expr::App(Box::new(foo_expr.clone()), vec![n_ref, foo_expr]);

        let main_func = Function::new(main_ident,vec![],vec![n_ident], n_assign, foo_call);
        let parsed = SyntaxTree::new(vec![foo_func,main_func]);

        assert_eq!(parse(&prog).unwrap(),parsed);
    }

      #[test]
    fn simple_records() {
        let prog = "foo() {
	var x;
	x={n:{foo:1,bar:2}};
	y = n.x.foo==n.x.bar;
	return y;
}
";
		let x_ident = Identifier::new("x".to_owned());
		let foo_ident = Identifier::new("foo".to_owned());
		let bar_ident = Identifier::new("bar".to_owned());
		let n_ident = Identifier::new("n".to_owned());
		let y_ident = Identifier::new("y".to_owned());
		let mut inner_rec_map = HashMap::new();
		inner_rec_map.insert(foo_ident.clone(), Expr::Value(Literal::Number(1)));
		inner_rec_map.insert(bar_ident.clone(), Expr::Value(Literal::Number(2)));
		let inner_record = Expr::Record(inner_rec_map);
		let mut outer_record_map = HashMap::new();
		outer_record_map.insert(n_ident.clone(), inner_record);

		let outer_record = Expr::Record(outer_record_map);
		let x_assign = Stmt::Assignment(x_ident.clone(), outer_record);
		let n_expr = Expr::Value(Literal::Identifier(n_ident.clone()));
		let n_by_x = Expr::Projection(Box::new(n_expr), x_ident.clone());
		let x_by_foo = Expr::Projection(Box::new(n_by_x.clone()), foo_ident.clone());
		let x_by_bar = Expr::Projection(Box::new(n_by_x), bar_ident);
		let comp = Expr::BinOp(BinaryOp::Equal, Box::new(x_by_foo), Box::new(x_by_bar));
		let y_assign = Stmt::Assignment(y_ident.clone(), comp);

		let y_expr = Expr::Value(Literal::Identifier(y_ident.clone()));
		let foo_func = Function::new(foo_ident, vec![], vec![x_ident], Stmt::Join(Box::new(x_assign),Box::new(y_assign)), y_expr);
		let parsed = SyntaxTree::new(vec![foo_func]);

		assert_eq!(parse(prog).unwrap(), parsed);
	}
}
