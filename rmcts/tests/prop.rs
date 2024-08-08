#![allow(unused_variables)]
// sources: egg/tests/prop.rs and rmcts/tests/math.rs

use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use std::collections::HashMap;

use egg::*;
use rmcts::run::{run_mcts, MCTSArgs};
use std::path::PathBuf;

define_language! {
    pub enum Prop {
        Bool(bool),
        "&" = And([Id; 2]),
        "~" = Not(Id),
        "|" = Or([Id; 2]),
        "->" = Implies([Id; 2]),
        Symbol(Symbol),
    }
}

type EGraph = egg::EGraph<Prop, ConstantFold>;
type Rewrite = egg::Rewrite<Prop, ConstantFold>;

#[derive(Default, Clone)]
pub struct ConstantFold;
impl Analysis<Prop> for ConstantFold {
    type Data = Option<(bool, PatternAst<Prop>)>;
    fn merge(&mut self, to: &mut Self::Data, from: Self::Data) -> DidMerge {
        merge_option(to, from, |a, b| {
            assert_eq!(a.0, b.0, "Merged non-equal constants");
            DidMerge(false, false)
        })
    }

    fn make(egraph: &EGraph, enode: &Prop) -> Self::Data {
        let x = |i: &Id| egraph[*i].data.as_ref().map(|c| c.0);
        let result = match enode {
            Prop::Bool(c) => Some((*c, c.to_string().parse().unwrap())),
            Prop::Symbol(_) => None,
            Prop::And([a, b]) => Some((
                x(a)? && x(b)?,
                format!("(& {} {})", x(a)?, x(b)?).parse().unwrap(),
            )),
            Prop::Not(a) => Some((!x(a)?, format!("(~ {})", x(a)?).parse().unwrap())),
            Prop::Or([a, b]) => Some((
                x(a)? || x(b)?,
                format!("(| {} {})", x(a)?, x(b)?).parse().unwrap(),
            )),
            Prop::Implies([a, b]) => Some((
                !x(a)? || x(b)?,
                format!("(-> {} {})", x(a)?, x(b)?).parse().unwrap(),
            )),
        };
        // println!("Make: {:?} -> {:?}", enode, result);
        result
    }

    fn modify(egraph: &mut EGraph, id: Id) {
        if let Some(c) = egraph[id].data.clone() {
            egraph.union_instantiations(
                &c.1,
                &c.0.to_string().parse().unwrap(),
                &Default::default(),
                "analysis".to_string(),
            );
        }
    }
}

macro_rules! rule {
    ($name:ident, $left:literal, $right:literal) => {
        #[allow(dead_code)]
        fn $name() -> Rewrite {
            rewrite!(stringify!($name); $left => $right)
        }
    };
    ($name:ident, $name2:ident, $left:literal, $right:literal) => {
        rule!($name, $left, $right);
        rule!($name2, $right, $left);
    };
}

rule! {def_imply, def_imply_flip,   "(-> ?a ?b)",       "(| (~ ?a) ?b)"          }
rule! {double_neg, double_neg_flip,  "(~ (~ ?a))",       "?a"                     }
rule! {assoc_or,    "(| ?a (| ?b ?c))", "(| (| ?a ?b) ?c)"       }
rule! {dist_and_or, "(& ?a (| ?b ?c))", "(| (& ?a ?b) (& ?a ?c))"}
rule! {dist_or_and, "(| ?a (& ?b ?c))", "(& (| ?a ?b) (| ?a ?c))"}
rule! {comm_or,     "(| ?a ?b)",        "(| ?b ?a)"              }
rule! {comm_and,    "(& ?a ?b)",        "(& ?b ?a)"              }
rule! {lem,         "(| ?a (~ ?a))",    "true"                      }
rule! {or_true,     "(| ?a true)",         "true"                      }
rule! {and_true,    "(& ?a true)",         "?a"                     }
rule! {contrapositive, "(-> ?a ?b)",    "(-> (~ ?b) (~ ?a))"     }

// this has to be a multipattern since (& (-> ?a ?b) (-> (~ ?a) ?c))  !=  (| ?b ?c)
// see https://github.com/egraphs-good/egg/issues/185
fn lem_imply() -> Rewrite {
    multi_rewrite!(
        "lem_imply";
        "?value = true = (& (-> ?a ?b) (-> (~ ?a) ?c))"
        =>
        "?value = (| ?b ?c)"
    )
}

#[rustfmt::skip]
pub fn rules() -> Vec<Rewrite> { vec![
    rewrite!("def_imply";  "(-> ?a ?b)"        => "(| (~ ?a) ?b)"),
    rewrite!("double_neg";  "(~ (~ ?a))"        => "?a"),
    rewrite!("assoc_or";  "(| ?a (| ?b ?c))"        => "(| (| ?a ?b) ?c)"),
    rewrite!("dist_and_or";  "(& ?a (| ?b ?c))"        => "(| (& ?a ?b) (& ?a ?c))"),
    rewrite!("dist_or_and";  "(| ?a (& ?b ?c))"        => "(& (| ?a ?b) (| ?a ?c))"),
    rewrite!("comm_or";  "(| ?a ?b)"        => "(| ?b ?a)"),
    rewrite!("comm_and";  "(& ?a ?b)"        => "(& ?b ?a)"),
    rewrite!("lem";  "(| ?a (~ ?a))"        => "true"),
    rewrite!("or_true";  "(| ?a true)"        => "true"),
    rewrite!("and_true";  "(& ?a true)"        => "?a"),
    rewrite!("contrapositive";  "(-> ?a ?b)"        => "(-> (~ ?b) (~ ?a))"),
    multi_rewrite!("lem_imply";  "?value = true = (& (-> ?a ?b) (-> (~ ?a) ?c))"        => "?value = (| ?b ?c)"),
]}

fn prove_something(name: &str, start: &str, rewrites: &[Rewrite], goals: &[&str]) {
    let _ = env_logger::builder().is_test(true).try_init();
    println!("Proving {}", name);

    let start_expr: RecExpr<_> = start.parse().unwrap();
    let goal_exprs: Vec<RecExpr<_>> = goals.iter().map(|g| g.parse().unwrap()).collect();

    let mut runner = Runner::default()
        .with_iter_limit(20)
        .with_node_limit(5_000)
        .with_expr(&start_expr);

    // we are assume the input expr is true
    // this is needed for the soundness of lem_imply
    let true_id = runner.egraph.add(Prop::Bool(true));
    let root = runner.roots[0];
    runner.egraph.union(root, true_id);
    runner.egraph.rebuild();

    let egraph = runner.run(rewrites).egraph;

    for (i, (goal_expr, goal_str)) in goal_exprs.iter().zip(goals).enumerate() {
        println!("Trying to prove goal {}: {}", i, goal_str);
        let equivs = egraph.equivs(&start_expr, goal_expr);
        if equivs.is_empty() {
            panic!("Couldn't prove goal {}: {}", i, goal_str);
        }
    }
}

#[test]
fn prove_contrapositive() {
    let _ = env_logger::builder().is_test(true).try_init();
    let rules = &[def_imply(), def_imply_flip(), double_neg_flip(), comm_or()];
    prove_something(
        "contrapositive",
        "(-> x y)",
        rules,
        &[
            "(-> x y)",
            "(| (~ x) y)",
            "(| (~ x) (~ (~ y)))",
            "(| (~ (~ y)) (~ x))",
            "(-> (~ y) (~ x))",
        ],
    );
}

#[test]
fn prove_chain() {
    let _ = env_logger::builder().is_test(true).try_init();
    let rules = &[
        // rules needed for contrapositive
        def_imply(),
        def_imply_flip(),
        double_neg_flip(),
        comm_or(),
        // and some others
        comm_and(),
        lem_imply(),
    ];
    prove_something(
        "chain",
        "(& (-> x y) (-> y z))",
        rules,
        &[
            "(& (-> x y) (-> y z))",
            "(& (-> (~ y) (~ x)) (-> y z))",
            "(& (-> y z)         (-> (~ y) (~ x)))",
            "(| z (~ x))",
            "(| (~ x) z)",
            "(-> x z)",
        ],
    );
}

#[test]
fn const_fold() {
    let start = "(| (& false true) (& true false))";
    let start_expr = start.parse().unwrap();
    let end = "false";
    let end_expr = end.parse().unwrap();
    let mut eg = EGraph::default();
    eg.add_expr(&start_expr);
    eg.rebuild();
    assert!(!eg.equivs(&start_expr, &end_expr).is_empty());
}

fn build_rand_expr(seed: u64, depth: u32) -> RecExpr<Prop> {
    const OPS: [&str; 4] = [
        "&", "~", "|", "->",
    ];
    const SYM: &str = "a";
    const BOOLS: [bool; 2] = [false, true];
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let op2children = HashMap::from([
        ("&", 2),
        ("~", 1),
        ("|", 2),
        ("->", 2),
    ]);
    let mut expr = RecExpr::default();
    dfs(depth, &mut expr, &OPS, &SYM, &BOOLS, &mut rng, &op2children);
    expr
}

fn dfs(
    depth: u32,
    expr: &mut RecExpr<Prop>,
    ops: &[&str],
    sym: &str,
    bools: &[bool],
    rng: &mut ChaCha8Rng,
    op2children: &HashMap<&str, u32>,
) -> egg::Id {
    if depth == 0 {
        // term
        let leaf;
        let rand = rng.gen_range(0..3);
        if rand < 2 {
            leaf = Prop::Bool(bools[rand]);
        } else {
            leaf = Prop::Symbol(sym.into());
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
            let id = dfs(depth - 1, expr, ops, sym, bools, rng, op2children);
            ids.push(id);
        }

        let node = Prop::from_op(op, ids).unwrap();
        let id = expr.add(node);
        return id;
    }
}

#[test]
fn prop_egg() {
    // build
    let depth = 10;
    let seed = 1;
    let expr = build_rand_expr(seed, depth);
    run_egg(true, &expr);
    run_egg(false, &expr);

    fn run_egg(backoff: bool, expr: &RecExpr<Prop>) {
        let runner = if backoff {
            Runner::default()
                .with_iter_limit(100)
                .with_node_limit(1_000)
                .with_expr(expr)
        } else {
            Runner::default()
                .with_iter_limit(100)
                .with_node_limit(1_000)
                .with_scheduler(egg::SimpleScheduler)
                .with_expr(expr)
        };
        let root = runner.roots[0];
        // base cost
        let extractor = Extractor::new(&runner.egraph, AstSize);
        let (base_cost, _base) = extractor.find_best(root);
        // best
        let runner = runner.run(&rules());
        let extractor = Extractor::new(&runner.egraph, AstSize);
        let (best_cost, best) = extractor.find_best(root);
        println!(
            "Simplified {} to {} with base_cost {} -> cost {}",
            expr, best, base_cost, best_cost
        );
        runner.print_report();
    }
}

#[test]
fn prop_mcts_geb() {
    println!("num rules {}", rules().len());
    // build
    let depth = 10;
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
        output_dir: PathBuf::from("/usr/experiments/tests/"),
        // egg
        node_limit: 1_000,
        time_limit: 10,
    };
    run_mcts(runner.egraph, root, rules(), AstSize, Some(args));
}