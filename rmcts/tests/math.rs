#![allow(unused_variables)]

use egg::{rewrite as rw, *};
use ordered_float::NotNan;
use rmcts::run::{run_mcts, MCTSArgs};

use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use std::collections::HashMap;
use std::path::PathBuf;

pub type EGraph = egg::EGraph<Math, ConstantFold>;
pub type Rewrite = egg::Rewrite<Math, ConstantFold>;

pub type Constant = NotNan<f64>;

define_language! {
    pub enum Math {
        "d" = Diff([Id; 2]),
        "i" = Integral([Id; 2]),

        "+" = Add([Id; 2]),
        "-" = Sub([Id; 2]),
        "*" = Mul([Id; 2]),
        "/" = Div([Id; 2]),
        "pow" = Pow([Id; 2]),
        "ln" = Ln(Id),
        "sqrt" = Sqrt(Id),

        "sin" = Sin(Id),
        "cos" = Cos(Id),

        Constant(Constant),
        Symbol(Symbol), }
}

// You could use egg::AstSize, but this is useful for debugging, since
// it will really try to get rid of the Diff operator
#[derive(Clone)]
pub struct MathCostFn;
impl egg::CostFunction<Math> for MathCostFn {
    type Cost = usize;
    fn cost<C>(&mut self, enode: &Math, mut costs: C) -> Self::Cost
    where
        C: FnMut(Id) -> Self::Cost,
    {
        // only SymbolLang provides `op` field, which can `as_str()`
        // let op_cost = match enode.op.as_str() {
        //     "d" => 100,
        //     "i" => 100,
        //     _ => 1,
        // };
        // enode.fold(op_cost, |sum, i| sum + costs(i))
        //
        let op_cost = match enode {
            Math::Diff(..) => 100,
            Math::Integral(..) => 100,
            _ => 1,
        };
        enode.fold(op_cost, |sum, i| sum + costs(i))
    }
}

#[cfg_attr(docsrs, doc(cfg(feature = "lp")))]
impl LpCostFunction<Math, ConstantFold> for MathCostFn {
    fn node_cost(&mut self, _egraph: &EGraph, _eclass: Id, _enode: &Math) -> f64 {
        1.0
    }
}

#[derive(Default, Clone)]
pub struct ConstantFold;
impl Analysis<Math> for ConstantFold {
    type Data = Option<(Constant, PatternAst<Math>)>;

    fn make(egraph: &EGraph, enode: &Math) -> Self::Data {
        let x = |i: &Id| egraph[*i].data.as_ref().map(|d| d.0);
        Some(match enode {
            Math::Constant(c) => (*c, format!("{}", c).parse().unwrap()),
            Math::Add([a, b]) => (
                x(a)? + x(b)?,
                format!("(+ {} {})", x(a)?, x(b)?).parse().unwrap(),
            ),
            Math::Sub([a, b]) => (
                x(a)? - x(b)?,
                format!("(- {} {})", x(a)?, x(b)?).parse().unwrap(),
            ),
            Math::Mul([a, b]) => (
                x(a)? * x(b)?,
                format!("(* {} {})", x(a)?, x(b)?).parse().unwrap(),
            ),
            Math::Div([a, b]) if x(b) != Some(NotNan::new(0.0).unwrap()) => (
                x(a)? / x(b)?,
                format!("(/ {} {})", x(a)?, x(b)?).parse().unwrap(),
            ),
            _ => return None,
        })
    }

    fn merge(&mut self, to: &mut Self::Data, from: Self::Data) -> DidMerge {
        merge_option(to, from, |a, b| {
            assert_eq!(a.0, b.0, "Merged non-equal constants");
            DidMerge(false, false)
        })
    }

    fn modify(egraph: &mut EGraph, id: Id) {
        let data = egraph[id].data.clone();
        if let Some((c, pat)) = data {
            if egraph.are_explanations_enabled() {
                egraph.union_instantiations(
                    &pat,
                    &format!("{}", c).parse().unwrap(),
                    &Default::default(),
                    "constant_fold".to_string(),
                );
            } else {
                let added = egraph.add(Math::Constant(c));
                egraph.union(id, added);
            }
            // to not prune, comment this out
            egraph[id].nodes.retain(|n| n.is_leaf());

            #[cfg(debug_assertions)]
            egraph[id].assert_unique_leaves();
        }
    }
}

fn is_const_or_distinct_var(v: &str, w: &str) -> impl Fn(&mut EGraph, Id, &Subst) -> bool {
    let v = v.parse().unwrap();
    let w = w.parse().unwrap();
    move |egraph, _, subst| {
        egraph.find(subst[v]) != egraph.find(subst[w])
            && (egraph[subst[v]].data.is_some()
                || egraph[subst[v]]
                    .nodes
                    .iter()
                    .any(|n| matches!(n, Math::Symbol(..))))
    }
}

fn is_const(var: &str) -> impl Fn(&mut EGraph, Id, &Subst) -> bool {
    let var = var.parse().unwrap();
    move |egraph, _, subst| egraph[subst[var]].data.is_some()
}

fn is_sym(var: &str) -> impl Fn(&mut EGraph, Id, &Subst) -> bool {
    let var = var.parse().unwrap();
    move |egraph, _, subst| {
        egraph[subst[var]]
            .nodes
            .iter()
            .any(|n| matches!(n, Math::Symbol(..)))
    }
}

fn is_not_zero(var: &str) -> impl Fn(&mut EGraph, Id, &Subst) -> bool {
    let var = var.parse().unwrap();
    move |egraph, _, subst| {
        if let Some(n) = &egraph[subst[var]].data {
            *(n.0) != 0.0
        } else {
            true
        }
    }
}

#[rustfmt::skip]
pub fn rules() -> Vec<Rewrite> { vec![
    rw!("comm-add";  "(+ ?a ?b)"        => "(+ ?b ?a)"),
    rw!("comm-mul";  "(* ?a ?b)"        => "(* ?b ?a)"),
    rw!("assoc-add"; "(+ ?a (+ ?b ?c))" => "(+ (+ ?a ?b) ?c)"),
    rw!("assoc-mul"; "(* ?a (* ?b ?c))" => "(* (* ?a ?b) ?c)"),

    rw!("sub-canon"; "(- ?a ?b)" => "(+ ?a (* -1 ?b))"),
    rw!("div-canon"; "(/ ?a ?b)" => "(* ?a (pow ?b -1))" if is_not_zero("?b")),
    // rw!("canon-sub"; "(+ ?a (* -1 ?b))"   => "(- ?a ?b)"),
    // rw!("canon-div"; "(* ?a (pow ?b -1))" => "(/ ?a ?b)" if is_not_zero("?b")),

    rw!("zero-add"; "(+ ?a 0)" => "?a"),
    rw!("zero-mul"; "(* ?a 0)" => "0"),
    rw!("one-mul";  "(* ?a 1)" => "?a"),

    rw!("add-zero"; "?a" => "(+ ?a 0)"),
    rw!("mul-one";  "?a" => "(* ?a 1)"),

    rw!("cancel-sub"; "(- ?a ?a)" => "0"),
    rw!("cancel-div"; "(/ ?a ?a)" => "1" if is_not_zero("?a")),

    rw!("distribute"; "(* ?a (+ ?b ?c))"        => "(+ (* ?a ?b) (* ?a ?c))"),
    rw!("factor"    ; "(+ (* ?a ?b) (* ?a ?c))" => "(* ?a (+ ?b ?c))"),

    rw!("pow-mul"; "(* (pow ?a ?b) (pow ?a ?c))" => "(pow ?a (+ ?b ?c))"),
    rw!("pow0"; "(pow ?x 0)" => "1"
        if is_not_zero("?x")),
    rw!("pow1"; "(pow ?x 1)" => "?x"),
    rw!("pow2"; "(pow ?x 2)" => "(* ?x ?x)"),
    rw!("pow-recip"; "(pow ?x -1)" => "(/ 1 ?x)"
        if is_not_zero("?x")),
    rw!("recip-mul-div"; "(* ?x (/ 1 ?x))" => "1" if is_not_zero("?x")),

    rw!("d-variable"; "(d ?x ?x)" => "1" if is_sym("?x")),
    rw!("d-constant"; "(d ?x ?c)" => "0" if is_sym("?x") if is_const_or_distinct_var("?c", "?x")),

    rw!("d-add"; "(d ?x (+ ?a ?b))" => "(+ (d ?x ?a) (d ?x ?b))"),
    rw!("d-mul"; "(d ?x (* ?a ?b))" => "(+ (* ?a (d ?x ?b)) (* ?b (d ?x ?a)))"),

    rw!("d-sin"; "(d ?x (sin ?x))" => "(cos ?x)"),
    rw!("d-cos"; "(d ?x (cos ?x))" => "(* -1 (sin ?x))"),

    rw!("d-ln"; "(d ?x (ln ?x))" => "(/ 1 ?x)" if is_not_zero("?x")),

    rw!("d-power";
        "(d ?x (pow ?f ?g))" =>
        "(* (pow ?f ?g)
            (+ (* (d ?x ?f)
                  (/ ?g ?f))
               (* (d ?x ?g)
                  (ln ?f))))"
        if is_not_zero("?f")
        if is_not_zero("?g")
    ),

    rw!("i-one"; "(i 1 ?x)" => "?x"),
    rw!("i-power-const"; "(i (pow ?x ?c) ?x)" =>
        "(/ (pow ?x (+ ?c 1)) (+ ?c 1))" if is_const("?c")),
    rw!("i-cos"; "(i (cos ?x) ?x)" => "(sin ?x)"),
    rw!("i-sin"; "(i (sin ?x) ?x)" => "(* -1 (cos ?x))"),
    rw!("i-sum"; "(i (+ ?f ?g) ?x)" => "(+ (i ?f ?x) (i ?g ?x))"),
    rw!("i-dif"; "(i (- ?f ?g) ?x)" => "(- (i ?f ?x) (i ?g ?x))"),
    rw!("i-parts"; "(i (* ?a ?b) ?x)" =>
        "(- (* ?a (i ?b ?x)) (i (* (d ?x ?a) (i ?b ?x)) ?x))"),
]}

#[test]
fn math_build_lang_by_hand() {
    let mut expr = RecExpr::default();

    let leaf1 = Math::Symbol("a".into());
    let leaf2 = Math::Constant(NotNan::new(1.0).unwrap());
    let id1 = expr.add(leaf1);
    let id2 = expr.add(leaf2);

    let node = Math::from_op("*", vec![id1, id2]).unwrap();
    expr.add(node);
    println!("is-dag {}", expr.is_dag());

    // run
    let runner = Runner::default()
        .with_iter_limit(20)
        .with_expr(&expr)
        .run(&rules());
    let root = runner.roots[0];
    let extractor = Extractor::new(&runner.egraph, AstSize);
    let (best_cost, best) = extractor.find_best(root);
    println!(
        "Simplified {} to {} with best cost {}",
        expr, best, best_cost
    );
}

#[test]
fn math_cost_fn() {
    let mut expr = RecExpr::default();

    let leaf1 = Math::Symbol("a".into());
    let leaf2 = Math::Constant(NotNan::new(1.0).unwrap());
    let id1 = expr.add(leaf1);
    let id2 = expr.add(leaf2);

    let node = Math::from_op("*", vec![id1, id2]).unwrap();
    expr.add(node);
}

#[test]
fn math_rand_seed() {
    let seed = 0;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let rand = rng.gen_range(0..100);

    for _ in 0..100 {
        let seed = 0;
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let sample = rng.gen_range(0..100);
        assert_eq!(rand, sample);
    }
}

fn build_rand_expr(seed: u64, depth: u32) -> RecExpr<Math> {
    const OPS: [&str; 11] = [
        "d", "i", "+", "-", "*", "/", "pow", "ln", "sqrt", "sin", "cos",
    ];
    const SYM: &str = "a";
    const NUM: [f64; 3] = [0.0, 1.0, 2.0];
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let op2children = HashMap::from([
        ("d", 2),
        ("i", 2),
        ("+", 2),
        ("-", 2),
        ("*", 2),
        ("/", 2),
        ("pow", 2),
        ("ln", 1),
        ("sqrt", 1),
        ("sin", 1),
        ("cos", 1),
    ]);
    let mut expr = RecExpr::default();
    dfs(depth, &mut expr, &OPS, &SYM, &NUM, &mut rng, &op2children);
    expr
}

fn dfs(
    depth: u32,
    expr: &mut RecExpr<Math>,
    ops: &[&str],
    sym: &str,
    num: &[f64],
    rng: &mut ChaCha8Rng,
    op2children: &HashMap<&str, u32>,
) -> egg::Id {
    if depth == 0 {
        // term
        let leaf;
        let rand = rng.gen_range(0..4);
        if rand < 3 {
            leaf = Math::Constant(NotNan::new(num[rand]).unwrap());
        } else {
            leaf = Math::Symbol(sym.into());
        }
        let id = expr.add(leaf);
        return id;
    } else {
        // op
        let rand = rng.gen_range(0..ops.len());
        let op = ops[rand];
        let n_child = op2children.get(&op).unwrap();
        let mut ids = vec![];
        for _ in 0..*n_child {
            let id = dfs(depth - 1, expr, ops, sym, num, rng, op2children);
            ids.push(id);
        }

        let node = Math::from_op(op, ids).unwrap();
        let id = expr.add(node);
        return id;
    }
}

#[test]
fn math_lp_extract() {
    let expr: RecExpr<Math> = "(pow (+ x (+ x x)) (+ x x))".parse().unwrap();

    let runner: Runner<Math, ConstantFold> = Runner::default()
        .with_iter_limit(3)
        .with_expr(&expr)
        .run(&rules());
    let root = runner.roots[0];

    let best = Extractor::new(&runner.egraph, AstSize).find_best(root).1;
    let lp_best = LpExtractor::new(&runner.egraph, AstSize).solve(root);

    println!("input   [{}] {}", expr.as_ref().len(), expr);
    println!("normal  [{}] {}", best.as_ref().len(), best);
    println!("ilp cse [{}] {}", lp_best.as_ref().len(), lp_best);

    assert_ne!(best, lp_best);
    assert_eq!(lp_best.as_ref().len(), 4);
}

#[test]
fn math_egg() {
    // build
    let depth = 7;
    let seed = 1;
    let expr = build_rand_expr(seed, depth);
    run_egg(true, &expr);
    run_egg(false, &expr);

    fn run_egg(backoff: bool, expr: &RecExpr<Math>) {
        let runner = if backoff {
            Runner::default().with_iter_limit(100).with_expr(expr)
        } else {
            Runner::default()
                .with_iter_limit(100)
                .with_scheduler(egg::SimpleScheduler)
                .with_expr(expr)
        };
        let root = runner.roots[0];
        // base cost
        // let extractor = Extractor::new(&runner.egraph, AstSize);
        let extractor = Extractor::new(&runner.egraph, MathCostFn);
        let (base_cost, _base) = extractor.find_best(root);
        // best
        let runner = runner.run(&rules());
        // let extractor = Extractor::new(&runner.egraph, AstSize);
        let extractor = Extractor::new(&runner.egraph, MathCostFn);
        let (best_cost, best) = extractor.find_best(root);
        println!(
            "Simplified {} to {} with base_cost {} -> cost {}",
            expr, best, base_cost, best_cost
        );
        runner.print_report();
    }
}

#[test]
fn math_mcts_geb() {
    println!("num rules {}", rules().len());
    // build
    let depth = 7;
    let seed = 1;
    let expr = build_rand_expr(seed, depth);
    let runner = Runner::default().with_expr(&expr);
    let root = runner.roots[0];
    let n_threads = std::thread::available_parallelism().unwrap().get();
    let args = MCTSArgs {
        // mcts
        budget: 512,
        max_sim_step: 10,
        gamma: 0.99,
        expansion_worker_num: 1,
        simulation_worker_num: 1, // n_threads - 1,
        lp_extract: false,
        cost_threshold: 1,
        iter_limit: 100,
        prune_actions: true,
        rollout_strategy: String::from("heavy"),
        subtree_caching: false,
        select_max_uct_action: false,
        // experiment tracking
        output_dir: PathBuf::from("usr/experiments/tests/"),
        // egg
        node_limit: 1_000,
        time_limit: 10,
    };
    run_mcts(runner.egraph, root, rules(), MathCostFn, Some(args));
}

#[test]
#[ignore]
fn math_mcts_geb_lp() {
    println!("num rules {}", rules().len());
    // build
    let depth = 7;
    let seed = 2;
    let expr = build_rand_expr(seed, depth);
    let runner = Runner::default().with_expr(&expr);
    let root = runner.roots[0];
    let n_threads = std::thread::available_parallelism().unwrap().get();
    let args = MCTSArgs {
        // mcts
        budget: 512,
        max_sim_step: 10,
        gamma: 0.99,
        expansion_worker_num: 1,
        simulation_worker_num: n_threads - 1,
        lp_extract: true,
        cost_threshold: 1,
        iter_limit: 30,
        prune_actions: false,
        rollout_strategy: String::from("random"),
        subtree_caching: false,
        select_max_uct_action: true,
        // experiment tracking
        output_dir: PathBuf::from("usr/experiments/tests/"),
        // egg
        node_limit: 500,
        time_limit: 10,
    };
    run_mcts(runner.egraph, root, rules(), MathCostFn, Some(args));
}
