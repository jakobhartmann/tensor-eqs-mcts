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

pub fn build_rand_expr(seed: u64, depth: u32) -> RecExpr<Prop> {
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