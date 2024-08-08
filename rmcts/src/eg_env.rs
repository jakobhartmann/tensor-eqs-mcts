#![allow(unused_variables)]
#![allow(unused_must_use)]

use std::time::Duration;
use std::fmt;
use std::path::{Path, PathBuf};
use std::hash::{Hash, Hasher};
use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use itertools::enumerate;
use egg::{
    EGraph, Extractor, Id, LpExtractor, Rewrite,
    Runner, SimpleScheduler, StopReason, RecExpr, Searcher
};
use tensat::model::{Mdl, TensorAnalysis};
use tensat::utils::extract_by_ilp_rmcts;
use tensat::optimize::{CostModel, TensorCost};
use tensat::rewrites::MultiPatterns;
use crate::env::Info;

#[derive(Clone)]
pub struct Ckpt {
    pub cnt: u32,
    pub sat_counter: usize,
    pub egraph: EGraph<Mdl, TensorAnalysis>,
    pub root_id: Id,
    pub last_cost: f32,
}

impl fmt::Display for Ckpt {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Cnt: {}, sat counter: {}, root id: {}, last cost: {}", self.cnt, self.sat_counter, self.root_id, self.last_cost)
    }
}

pub struct EgraphEnv {
    init_egraph: EGraph<Mdl, TensorAnalysis>,
    pub egraph: EGraph<Mdl, TensorAnalysis>,
    pub root_id: Id,
    pub init_expr: RecExpr<Mdl>,

    num_rules: usize,
    rules: Vec<Rewrite<Mdl, TensorAnalysis>>,
    num_multi_patterns: usize,
    multi_patterns: Option<MultiPatterns>,

    prune_actions: bool,

    node_limit: usize,
    time_limit: std::time::Duration,

    cost_model: CostModel,
    extraction: String,
    cost_hashmap: HashMap<u64, f32>,

    pub base_cost: f32,
    pub last_cost: f32,
    cnt: u32,
    sat_counter: usize,

    order_var_int: bool,
    class_constraint: bool,
    no_order: bool,
    initial_with_greedy: bool,
    ilp_time_sec: usize,
    ilp_num_threads: usize,
    output_dir: PathBuf,
}

impl EgraphEnv {
    pub fn new(
        egraph: EGraph<Mdl, TensorAnalysis>,
        root_id: Id,
        rules: Vec<Rewrite<Mdl, TensorAnalysis>>,
        multi_patterns: Option<MultiPatterns>,
        all_weight_only: bool,
        extraction: String,
        prune_actions: bool,
        node_limit: usize,
        time_limit: usize,
        // ilp
        order_var_int: bool,
        class_constraint: bool,
        no_order: bool,
        initial_with_greedy: bool,
        ilp_time_sec: usize,
        ilp_num_threads: usize,
        output_dir: PathBuf,
    ) -> Self {
        // TensorCost
        let cost_model: CostModel = tensat::optimize::CostModel::with_setting(all_weight_only);

        // Base cost extraction
        let (base_cost, base_expr) = match extraction.as_str() {
            "egg_greedy" => {
                let tnsr_cost: TensorCost = tensat::optimize::TensorCost::new(&egraph, &cost_model, true);
                let (cost, expr) = Extractor::new(&egraph, tnsr_cost).find_best(root_id);
                (cost, expr)
            }
            "new_greedy" => {
                let tnsr_cost: TensorCost = tensat::optimize::TensorCost::new(&egraph, &cost_model, false);
                let (cost, expr) = Extractor::new(&egraph, tnsr_cost).find_best(root_id);
                (cost, expr)
            }
            "egg_ilp" => {
                let tnsr_cost: TensorCost = tensat::optimize::TensorCost::new(&egraph, &cost_model, true);
                let (cost, expr) = LpExtractor::new(&egraph, tnsr_cost).solve(root_id);
                (cost as f32, expr)
            }
            "tensat_ilp" => {
                let ilp_dir = Path::new(&output_dir).join("ilp").into_os_string().into_string().unwrap();
                let (expr, cost, duration) = extract_by_ilp_rmcts(&egraph, root_id, &cost_model, order_var_int, class_constraint, no_order, initial_with_greedy, ilp_time_sec, ilp_num_threads, ilp_dir, true);
                (cost, expr)
            }
            _ => {
                panic!("Extraction method not found!");
            }
        };

        //
        EgraphEnv {
            init_egraph: egraph,
            egraph: EGraph::default(),
            root_id: root_id,
            init_expr: base_expr,
            num_rules: rules.len(),
            rules: rules,
            num_multi_patterns: match &multi_patterns {
                Some(v) => { v.rules.len() }
                None => { 0 }
            },
            multi_patterns: multi_patterns,
            prune_actions: prune_actions,
            node_limit: node_limit,
            time_limit: Duration::from_secs(time_limit.try_into().unwrap()),

            cost_model: cost_model,
            extraction: extraction,
            base_cost: base_cost,
            last_cost: 0.0,
            cnt: 0,
            sat_counter: 0,

            cost_hashmap: HashMap::new(),

            order_var_int: order_var_int,
            class_constraint: class_constraint,
            no_order: no_order,
            initial_with_greedy: initial_with_greedy,
            ilp_time_sec: ilp_time_sec,
            ilp_num_threads: ilp_num_threads,
            output_dir: output_dir.clone(),
        }
    }

    pub fn reset(&mut self) {
        self.cnt = 0;
        self.sat_counter = 0;
        self.egraph = self.init_egraph.clone();
        self.last_cost = self.base_cost;
    }

    pub fn step(&mut self, action: usize) -> ((), f32, bool, Info) {
        // run egg
        let egraph = std::mem::take(&mut self.egraph);

        let runner: Runner<Mdl, TensorAnalysis>;
        if action < self.num_rules {
            let rule = vec![self.rules[action].clone()];

            runner = Runner::default()
                .with_egraph(egraph)
                .with_iter_limit(1)
                .with_node_limit(self.node_limit)
                .with_time_limit(self.time_limit)
                .with_scheduler(SimpleScheduler)
                .run(&rule);
        } else {
            let rule = vec![];

            let multi_patterns = self.multi_patterns.clone();
            let multi_pattern_action = action - self.num_rules;

            runner = Runner::default()
                .with_egraph(egraph)
                .with_expr(&self.init_expr)
                .with_iter_limit(1)
                .with_node_limit(self.node_limit)
                .with_time_limit(self.time_limit)
                .with_scheduler(SimpleScheduler)
                .with_hook(move |runner| multi_patterns.clone().unwrap().run_one(runner, Some(multi_pattern_action)))
                .run(&rule);
        }

        let report = runner.report();

        // reclaim the partial egraph
        self.egraph = runner.egraph;

        // let num_applications: usize = runner
        //     .iterations
        //     .iter()
        //     .map(|i| i.applied.values().sum::<usize>())
        //     .sum();

        // run extract
        // Hash egraph
        let mut hasher = DefaultHasher::new();
        &self.egraph.hash(&mut hasher);
        let hash = hasher.finish();

        // Best cost extraction
        let best_cost = match self.cost_hashmap.get(&hash) {
            Some(cost) => {
                *cost
            }
            None => {
                let best_cost = match self.extraction.as_str() {
                    "egg_greedy" => {
                        let tnsr_cost: TensorCost = tensat::optimize::TensorCost::new(&self.egraph, &self.cost_model, true);
                        let cost = Extractor::new(&self.egraph, tnsr_cost).find_best_cost(self.root_id);
                        cost
                    }
                    "new_greedy" => {
                        let tnsr_cost: TensorCost = tensat::optimize::TensorCost::new(&self.egraph, &self.cost_model, false);
                        let cost = Extractor::new(&self.egraph, tnsr_cost).find_best_cost(self.root_id);
                        cost
                    }
                    "egg_ilp" => {
                        let tnsr_cost: TensorCost = tensat::optimize::TensorCost::new(&self.egraph, &self.cost_model, true);
                        let (cost, _) = LpExtractor::new(&self.egraph, tnsr_cost).solve(self.root_id);
                        cost as f32
                    }
                    "tensat_ilp" => {
                        let ilp_dir = Path::new(&self.output_dir).join("ilp").into_os_string().into_string().unwrap();
                        let (expr, cost, duration) = extract_by_ilp_rmcts(&self.egraph, self.root_id, &self.cost_model, self.order_var_int, self.class_constraint, self.no_order, self.initial_with_greedy, self.ilp_time_sec, self.ilp_num_threads, ilp_dir, false);
                        cost
                    }
                    _ => {
                        panic!("Extraction method not found!");
                    }
                };
                self.cost_hashmap.insert(hash, best_cost);
                best_cost
            }
        };

        // compute transition
        self.cnt += 1;
        let mut done = false;
        match runner.stop_reason.as_ref().unwrap() {
            StopReason::NodeLimit(_) => {
                done = true;
                self.sat_counter = 0;
                // println!(
                //     "EGG NodeLimit {}s - {}s - {} - {} - {}",
                //     node_limit,
                //     report.total_time,
                //     report.egraph_nodes,
                //     report.egraph_classes,
                //     report.memo_size,
                // );
            }
            StopReason::TimeLimit(time) => {
                // TODO this indicates egraph is exploded?
                done = true;
                println!(
                    "EGG TimeLimit {}s - {}s - {} - {} - {}",
                    time,
                    report.total_time,
                    report.egraph_nodes,
                    report.egraph_classes,
                    report.memo_size,
                );
            }
            StopReason::Saturated => {
                // TODO sat_counter is enough to indicate saturation?
                self.sat_counter += 1;
                if self.sat_counter == (self.num_rules + self.num_multi_patterns) {
                    done = true;
                }
            }
            StopReason::IterationLimit(_) => self.sat_counter = 0,
            _ => self.sat_counter = 0,
        }
        // compute reward
        let reward = f32::max(self.last_cost - best_cost, 0.0);
        // let reward = self.last_cost - best_cost;
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
    //     let reward = std::cmp::max(self.last_cost - best_cost, 0);

    //     reward as f32
    // }

    pub fn get_action_space(&self) -> usize {
        self.num_rules + self.num_multi_patterns
    }

    pub fn checkpoint(&self) -> Ckpt {
        Ckpt {
            cnt: self.cnt,
            sat_counter: self.sat_counter,
            egraph: self.egraph.clone(),
            root_id: self.root_id.clone(),
            last_cost: self.last_cost,
        }
    }

    pub fn restore(&mut self, checkpoint_data: Ckpt) {
        self.cnt = checkpoint_data.cnt;
        self.sat_counter = checkpoint_data.sat_counter;
        self.egraph = checkpoint_data.egraph;
        self.root_id = checkpoint_data.root_id;
        self.last_cost = checkpoint_data.last_cost;
    }

    pub fn action_pruning(&mut self, mut children_saturated: Vec<bool>) -> (Vec<bool>, usize) {
        if self.prune_actions {
            if children_saturated.iter().filter(|x| **x).count() > 0 {
                panic!("At least one child is already saturated. Should this ever happen?");
            }

            // Create runner with egraph
            let egraph = std::mem::take(&mut self.egraph);
            let mut runner = Runner::default().with_egraph(egraph);

            // If the runner is not clean, rebuild the egraph
            if !runner.egraph.clean {
                runner.egraph.rebuild();
            }

            // Iterate over all single-pattern rewrite rules and check if the source pattern is found at least once in the egraph. IMPORTANT: Search with limit, otherwise we get an OOM for larger egraphs!
            // If the source pattern is not found, mark the child accordingly and increase the saturation counter of the environment
            for (i, rewrite) in self.rules.iter().enumerate() {
                if rewrite.searcher.search_with_limit(&runner.egraph, 1).len() == 0 {
                    if !children_saturated[i] {
                        children_saturated[i] = true;
                        self.sat_counter += 1;
                    }
                }
            }

            // If multi-pattern rules exist, check if both source patterns are found in the graph. IMPORTANT: Search with limit, otherwise we get an OOM for larger egraphs!
            // If one of the source patterns is not found, mark the child accordingly and increase the saturation counter of the environment
            match &self.multi_patterns {
                Some(multi_pattern) => {
                    for (i, (src1, src2, _, _, _)) in enumerate(&multi_pattern.rules) {
                        if (src1.search_with_limit(&runner.egraph, 1).len() == 0) || (src2.search_with_limit(&runner.egraph, 1).len() == 0) {
                            if !children_saturated[i + self.num_rules] {
                                children_saturated[i + self.num_rules] = true;
                                self.sat_counter += 1;
                            }
                        }
                    }
                }
                None => {}
            }

            // Reclaim the partial graph
            self.egraph = runner.egraph;
        }

        // Calculate the number of saturated children and return it together with the corresponding list
        let children_saturated_cnt = children_saturated.iter().filter(|x| **x).count();
        (children_saturated, children_saturated_cnt)
    }
}
