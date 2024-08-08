use egg::{
    Analysis, AstSize, EGraph, Extractor, Id, Language, RecExpr, Report, Rewrite, Runner,
    SimpleScheduler, StopReason,
};
// use std::collections::HashMap;
use std::time::Duration;

pub struct Info {
    pub report: Report,
    pub best_cost: f32,
}

#[allow(dead_code)]
pub struct Env<L: Language, N: Analysis<L>> {
    action_history: Vec<usize>,
    expr: RecExpr<L>,
    egraph: EGraph<L, N>,
    root_id: Id,
    num_rules: usize,
    rules: Vec<Rewrite<L, N>>,

    node_limit: usize,
    time_limit: std::time::Duration,

    pub base_cost: f32,
    pub last_cost: f32,
    cnt: u32,
    sat_counter: usize,
}

#[allow(dead_code)]
impl<L, N> Env<L, N>
where
    L: Language + 'static + egg::FromOp + std::marker::Send,
    N: Analysis<L> + Clone + 'static + std::default::Default + std::marker::Send,
    N::Data: Clone,
{
    pub fn new(
        expr: RecExpr<L>,
        rules: Vec<Rewrite<L, N>>,
        node_limit: usize,
        time_limit: usize,
    ) -> Self {
        // get base
        let runner: Runner<L, N> = Runner::default().with_expr(&expr);
        let (base_cost, _) = Extractor::new(&runner.egraph, AstSize).find_best(runner.roots[0]);
        let base_cost = base_cost as f32;
        Env {
            action_history: Vec::new(),
            expr: expr,
            egraph: EGraph::default(),
            root_id: Id::default(),
            num_rules: rules.len(),
            rules: rules,
            node_limit: node_limit,
            time_limit: Duration::from_secs(time_limit.try_into().unwrap()),

            base_cost: base_cost,
            last_cost: base_cost,
            cnt: 0,
            sat_counter: 0,
        }
    }

    pub fn reset(&mut self) {
        self.action_history.clear();
        self.cnt = 0;
        self.sat_counter = 0;
        self.egraph = EGraph::default();
        self.root_id = self.egraph.add_expr(&self.expr);
        self.last_cost = self.base_cost;
    }

    pub fn step(&mut self, action: usize) -> ((), f32, bool, Info) {
        // run egg
        let egraph = std::mem::take(&mut self.egraph);
        let rule = vec![self.rules[action].clone()];
        let runner: Runner<L, N> = Runner::default()
            .with_egraph(egraph)
            .with_iter_limit(1)
            .with_node_limit(self.node_limit)
            .with_time_limit(self.time_limit)
            .with_scheduler(SimpleScheduler)
            .run(&rule);
        let report = runner.report();

        // reclaim the partial egraph
        self.egraph = runner.egraph;

        // let num_applications: usize = runner
        //     .iterations
        //     .iter()
        //     .map(|i| i.applied.values().sum::<usize>())
        //     .sum();

        // run extract
        let extractor = Extractor::new(&self.egraph, AstSize);
        let (best_cost, _) = extractor.find_best(self.root_id);
        let best_cost = best_cost as f32;

        // compute transition
        self.cnt += 1;
        self.action_history.push(action);
        let mut done = false;
        match runner.stop_reason.as_ref().unwrap() {
            StopReason::NodeLimit(_) => {
                done = true;
                self.sat_counter = 0;
            }
            StopReason::TimeLimit(time) => {
                // TODO think about how this enables dealing with straggelers!
                panic!("egg time limit {}", time);
            }
            StopReason::Saturated => {
                self.sat_counter += 1;
                if self.sat_counter == (self.num_rules) {
                    done = true;
                }
            }
            StopReason::IterationLimit(_) => self.sat_counter = 0,
            _ => self.sat_counter = 0,
        }
        // let reward = std::cmp::max(self.last_cost - best_cost, 0); // TODO allow callback cost func
        let reward = f32::max(self.last_cost - best_cost, 0.0);
        self.last_cost = best_cost;
        let info = Info {
            report: report,
            best_cost: best_cost,
        };

        ((), (reward as f32), done, info)
    }

    // immediately extract and get reward
    // pub fn get_reward(&self) -> f32 {
    //     let extractor = Extractor::new(&self.egraph, AstSize);
    //     let (best_cost, _) = extractor.find_best(self.root_id);
    //     let reward = std::cmp::max(self.last_cost - best_cost, 0); // TODO allow callback cost func

    //     reward as f32
    // }

    pub fn get_action_space(&self) -> usize {
        self.num_rules
    }

    pub fn checkpoint(&self) -> Vec<usize> {
        self.action_history.clone()
    }

    pub fn restore(&mut self, checkpoint_data: Vec<usize>) {
        self.reset();
        for action in checkpoint_data.into_iter() {
            self.step(action);
        }
    }
}

#[cfg(test)]
mod test {
    #![allow(unused_imports)]
    use super::*;
    use egg::*;
    use std::thread::sleep;
    use std::time::Duration;

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

    #[test]
    fn test_add_expr_all_return_the_all_id() {
        let expr = "(+ 0 (* 1 foo))".parse().unwrap();
        let mut egraph = EGraph::default();
        let id = egraph.add_expr(&expr);
        egraph.rebuild();
        let new_id = egraph.add_expr(&expr);
        egraph.rebuild();
        assert_eq!(id, new_id);
        println!("ID {:?} - {:?}", id, new_id);

        for _ in 0..5 {
            let mut runner = Runner::default()
                .with_egraph(egraph)
                .with_iter_limit(1)
                .with_node_limit(10000)
                .with_time_limit(Duration::from_secs(10))
                .with_scheduler(SimpleScheduler)
                .run(&make_rules());

            runner.egraph.rebuild();
            let new_id = runner.egraph.add_expr(&expr);
            assert_ne!(id, new_id); // XXX add_expr gives different roots?
            println!("ID {:?} - {:?}", id, new_id);

            egraph = runner.egraph;
        }
    }
}
