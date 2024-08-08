#![allow(unused_imports, unused_variables)]
use egg::*;
use rmcts::*;

define_language! {
    enum SimpleLanguage {
        Num(i32),
        "+" = Add([Id; 2]),
        "*" = Mul([Id; 2]),
        Symbol(Symbol),
    }
}

fn make_rules() -> Vec<Rewrite<SimpleLanguage, ()>> {
    vec![
        rewrite!("commute-add"; "(+ ?a ?b)" => "(+ ?b ?a)"),
        rewrite!("commute-mul"; "(* ?a ?b)" => "(* ?b ?a)"),
        rewrite!("add-0"; "(+ ?a 0)" => "?a"),
        rewrite!("mul-0"; "(* ?a 0)" => "0"),
        rewrite!("mul-1"; "(* ?a 1)" => "?a"),
    ]
}

/// parse an expression, simplify it using egg, and pretty print it back out
fn simplify(s: &str) -> String {
    // parse the expression, the type annotation tells it which Language to use
    let expr: RecExpr<SimpleLanguage> = s.parse().unwrap();

    // simplify the expression using a Runner, which creates an e-graph with
    // the given expression and runs the given rules over it
    let runner = Runner::default().with_expr(&expr);
    let root = runner.roots[0];

    // base cost
    let extractor = Extractor::new(&runner.egraph, AstSize);
    let (base_cost, _base) = extractor.find_best(root);

    // best
    let runner = runner.run(&make_rules());
    let extractor = Extractor::new(&runner.egraph, AstSize);
    let (best_cost, best) = extractor.find_best(root);

    println!(
        "Simplified {} to {} with base_cost {} -> cost {}",
        expr, best, base_cost, best_cost
    );
    best.to_string()
}

// from egg, and we need Clone trait
#[derive(Debug, Clone)]
pub struct AstSize;
impl<L: Language> CostFunction<L> for AstSize {
    type Cost = usize;
    fn cost<C>(&mut self, enode: &L, mut costs: C) -> Self::Cost
    where
        C: FnMut(Id) -> Self::Cost,
    {
        enode.fold(1, |sum, id| sum.saturating_add(costs(id)))
    }
}

#[cfg_attr(docsrs, doc(cfg(feature = "lp")))]
impl<L: Language, N: Analysis<L>> LpCostFunction<L, N> for AstSize {
    fn node_cost(&mut self, _egraph: &EGraph<L, N>, _eclass: Id, _enode: &L) -> f64 {
        1.0
    }
}

#[test]
fn simple_egg_test() {
    assert_eq!(simplify("(* 0 42)"), "0");
    assert_eq!(simplify("(+ 0 (* 1 foo))"), "foo");
}

#[test]
fn simple_test() {
    let mut expr = RecExpr::default();
    let a = expr.add(SymbolLang::leaf("a"));
    let b = expr.add(SymbolLang::leaf("b"));
    let foo = expr.add(SymbolLang::new("foo", vec![a, b]));

    // we can do the same thing with an EGraph
    let mut egraph: EGraph<SymbolLang, ()> = Default::default();
    let a = egraph.add(SymbolLang::leaf("a"));
    let b = egraph.add(SymbolLang::leaf("b"));
    let foo = egraph.add(SymbolLang::new("foo", vec![a, b]));

    // we can also add RecExprs to an egraph
    let foo2 = egraph.add_expr(&expr);
    // note that if you add the same thing to an e-graph twice, you'll get back equivalent Ids
    assert_eq!(foo, foo2);

    let mut find = false;
    let enodes = expr.as_ref();
    for en in enodes.iter() {
        if en.op.as_str() == "foo" {
            find = true;
            println!("okokokoko");
        }
    }
    assert!(find);
}
