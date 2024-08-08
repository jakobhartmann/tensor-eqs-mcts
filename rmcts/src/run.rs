#![allow(unused_imports)]

use std::path::PathBuf;
use egg::{Analysis, CostFunction, EGraph, Id, Language, LpCostFunction, RecExpr, Rewrite};
use tensat::model::{Mdl, TensorAnalysis};
use tensat::rewrites::MultiPatterns;
use crate::tree;

pub struct MCTSArgs {
    // mcts
    pub budget: u32,
    pub max_sim_step: u32,
    pub gamma: f32,
    pub expansion_worker_num: usize,
    pub simulation_worker_num: usize,
    pub all_weight_only: bool,
    pub extraction: String,
    pub final_extraction: String,
    pub cost_threshold: f32,
    pub iter_limit: usize,
    pub prune_actions: bool,
    pub rollout_strategy: String,
    pub subtree_caching: bool,
    pub select_max_uct_action: bool,
    // egg
    pub node_limit: usize,
    pub time_limit: usize,
    // experiment tracking
    pub output_dir: PathBuf,
    pub save_graph: String,
    pub export_models: bool,
    // ilp
    pub order_var_int: bool,
    pub class_constraint: bool,
    pub no_order: bool,
    pub initial_with_greedy: bool,
    pub ilp_time_sec: usize,
    pub ilp_num_threads: usize,
}

pub fn run_mcts(
    egraph: EGraph<Mdl, TensorAnalysis>,
    id: Id,
    rules: Vec<Rewrite<Mdl, TensorAnalysis>>,
    multi_patterns: Option<MultiPatterns>,
    args: Option<MCTSArgs>,
) -> EGraph<Mdl, TensorAnalysis>
{
    // Args
    // mcts
    let mut budget = 512;
    let mut max_sim_step = 5;
    let mut gamma = 0.99;
    let mut expansion_worker_num = 1;
    let mut simulation_worker_num = 1;
    let mut all_weight_only = true;
    let mut extraction = String::from("egg_greedy");
    let mut final_extraction = String::from("egg_greedy");
    let mut cost_threshold = 0.1;
    let mut iter_limit = 200;
    let mut prune_actions = false;
    let mut rollout_strategy = String::from("random");
    let mut subtree_caching = false;
    let mut select_max_uct_action = true; // true -> max uct action; false -> max visited action
    // let verbose = false;
    // egg
    let mut node_limit = 2_000;
    let mut time_limit = 1;
    // experiment tracking
    let mut output_dir = PathBuf::new();
    let mut save_graph = String::from("none");
    let mut export_models = false;
    // ilp
    let mut order_var_int = false;
    let mut class_constraint = false;
    let mut no_order = false;
    let mut initial_with_greedy = false;
    let mut ilp_time_sec = 600;
    let mut ilp_num_threads = 1;


    // overwrite params if possible
    match args {
        None => (),
        Some(args) => {
            // mcts
            budget = args.budget;
            max_sim_step = args.max_sim_step;
            gamma = args.gamma;
            expansion_worker_num = args.expansion_worker_num;
            simulation_worker_num = args.simulation_worker_num;
            all_weight_only = args.all_weight_only;
            extraction = args.extraction;
            final_extraction = args.final_extraction;
            cost_threshold = args.cost_threshold;
            iter_limit = args.iter_limit;
            prune_actions = args.prune_actions;
            rollout_strategy = args.rollout_strategy;
            subtree_caching = args.subtree_caching;
            select_max_uct_action = args.select_max_uct_action;
            // egg
            node_limit = args.node_limit;
            time_limit = args.time_limit;
            // experiment tracking
            output_dir = args.output_dir;
            save_graph = args.save_graph;
            export_models = args.export_models;
            // ilp
            order_var_int = args.order_var_int;
            class_constraint = args.class_constraint;
            no_order = args.no_order;
            initial_with_greedy = args.initial_with_greedy;
            ilp_time_sec = args.ilp_time_sec;
            ilp_num_threads = args.ilp_num_threads;
        }
    }

    // Run
    let mut mcts = tree::Tree::new(
        // mcts
        budget,
        max_sim_step,
        gamma,
        expansion_worker_num,
        simulation_worker_num,
        prune_actions,
        rollout_strategy,
        subtree_caching,
        select_max_uct_action,
        // egg
        egraph.clone(),
        id.clone(),
        rules.clone(),
        multi_patterns.clone(),
        all_weight_only,
        extraction,
        final_extraction,
        node_limit,
        time_limit,
        // experiment tracking
        output_dir.clone(),
        save_graph,
        export_models,
        // ilp
        order_var_int,
        class_constraint,
        no_order,
        initial_with_greedy,
        ilp_time_sec,
        ilp_num_threads,
    );
    mcts.run_loop(egraph, id, rules.clone(), multi_patterns.clone(), cost_threshold, iter_limit)
}
