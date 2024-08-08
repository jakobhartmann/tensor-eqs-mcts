#![allow(unused)]
#![allow(unused_variables)]
#![allow(unused_must_use)]

mod eg_env;
mod env;
mod node;
mod pool_manager;
mod run;
mod tree;
mod workers;
mod math;
mod prop;
mod utils;
mod custom_models;

use std::fs::{create_dir_all, read_to_string};
use std::path::Path;
use std::time::Instant;
use serde::{Serialize, Deserialize};
use serde_json::{to_string, json};
use clap::{App, Arg};

use egg::*;
use tensat::utils::{save_model, extract_by_ilp_rmcts};
use tensat::model::{Mdl, TensorAnalysis};
use tensat::optimize::{CostModel, TensorCost};
use tensat::rewrites::MultiPatterns;
use tensat::{bert, inceptionv3, mobilenetv2, nasneta, nasrnn, resnet50, resnext50, squeezenet, vgg};
use crate::run::MCTSArgs;
use crate::utils::save_data_to_file;
use crate::custom_models::{get_very_simple_model, get_simple_model, get_complex_model, get_complex_model2, get_suboptimal_model, get_138_model};

use std::alloc;
use cap::Cap;
#[global_allocator]
static ALLOCATOR: Cap<alloc::System> = Cap::new(alloc::System, usize::max_value());

define_language! {
    enum SimpleLanguage {
        Num(i32),
        "+" = Add([Id; 2]),
        "*" = Mul([Id; 2]),
        Symbol(Symbol),
    }
}

fn make_rules() -> Vec<egg::Rewrite<SimpleLanguage, ()>> {
    vec![
        rewrite!("commute-add"; "(+ ?a ?b)" => "(+ ?b ?a)"),
        rewrite!("commute-mul"; "(* ?a ?b)" => "(* ?b ?a)"),
        rewrite!("add-0"; "(+ ?a 0)" => "?a"),
        rewrite!("mul-0"; "(* ?a 0)" => "0"),
        rewrite!("mul-1"; "(* ?a 1)" => "?a"),
    ]
}

// from egg, and we need Clone trait
#[derive(Debug, Clone, Serialize)]
pub struct AstSize;
impl<L: Language> CostFunction<L> for AstSize {
    type Cost = usize;
    fn cost<C>(&mut self, enode: &L, mut costs: C, _eclass_id: Option<Id>) -> Self::Cost
    where
        C: FnMut(Id) -> Self::Cost,
    {
        enode.fold(1, |sum, id| sum.saturating_add(costs(id)))
    }
}

#[cfg_attr(docsrs, doc(cfg(feature = "lp")))]
impl<L: Language, N: Analysis<L>> LpCostFunction<L, N> for AstSize {
    fn node_cost(&mut self, _egraph: &egg::EGraph<L, N>, _eclass: Id, _enode: &L) -> f64 {
        1.0
    }
}

#[derive(Clone, Serialize, Deserialize)]
struct Settings {
    model: String, // Model to optimize
    rules: String, // Provide a file with rewrite rules
    multi_rules: String, // File with multi-pattern rules. Every two lines belong to one multi-pattern rule
    save_graph: String, // all, io, none

    n_sec: u64, // time_limit: Max number of seconds for egg to run
    n_nodes: usize, // n_nodes: Max number of nodes for egraph
    ilp_time_sec: usize, // Time limit for ILP solver (0 = no limit)
    ilp_num_threads: usize, // Number of threads for ILP solver
    node_multi: usize, // Max number of nodes added by multi-pattern rules
    iter_multi: usize, // Max number of iterations to apply multi-pattern rules -> should always be 1 for RMCTS

    export_models: bool, // Whether or not to store input and optimized model
    use_multi: bool, // Set this flag will enable the use of multi-pattern rules
    no_order: bool, // No ordering constraints in ILP
    all_weight_only: bool, // Treat zero cost for all weight concat only
    no_cycle: bool, // Not allowing cycles in EGraph

    order_var_int: bool, // Set this flag will let ILP use integer var for ordering
    class_constraint: bool, // Add constraint in ILP that each eclass sum to 1
    initial_with_greedy: bool, // Initialize ILP with greedy solution
    filter_before: bool, // Filter cycles before applying rules

    // mcts
    budget: u32,
    max_sim_step: u32,
    gamma: f32,
    expansion_worker_num: usize,
    simulation_worker_num: usize,
    extraction: String, // egg_greedy, egg_ilp, ilp, new_greedy?
    final_extraction: String, // egg_greedy, egg_ilp, ilp, new_greedy?
    cost_threshold: f32,
    iter_limit: usize,
    prune_actions: bool,
    rollout_strategy: String,
    subtree_caching: bool,
    select_max_uct_action: bool,

    // experiment tracking
    seed: usize,
    experiments_base_path: String,
}

fn main() {
    // Read cmd arguments
    let args = App::new("tensor-eqs-mcts")
        .arg(
            Arg::with_name("model")
                .long("model")
                .takes_value(true)
                .default_value("138_model")
                .help("Model to optimize"),
        )
        .arg(
            Arg::with_name("seed")
                .long("seed")
                .takes_value(true)
                .default_value("0")
                .help("Seed for this experiment"),
        )
        .arg(
            Arg::with_name("extraction")
                .long("extraction")
                .takes_value(true)
                .default_value("new_greedy")
                .help("Main extraction method"),
        )
        .arg(
            Arg::with_name("final_extraction")
                .long("final_extraction")
                .takes_value(true)
                .default_value("tensat_ilp")
                .help("Final extraction method"),
        )
        .get_matches();

    let mut settings = Settings {
        model: String::from("resnet50"),
        rules: String::from("/usr/tensat/converted.txt"),
        multi_rules: String::from("/usr/tensat/converted_multi.txt"),
        save_graph: String::from("io"),

        n_sec: 10,
        n_nodes: 2_000,
        ilp_time_sec: 600,
        ilp_num_threads: 1,
        node_multi: 2_000,
        iter_multi: 1,

        export_models: true,
        use_multi: true,
        no_order: true,
        all_weight_only: true,
        no_cycle: true,

        filter_before: false,
        order_var_int: false,
        class_constraint: false,
        initial_with_greedy: false,

        // mcts
        budget: 128,
        max_sim_step: 10,
        gamma: 0.99, // used in rmcts?
        expansion_worker_num: 1,
        simulation_worker_num: 1,
        extraction: String::from("new_greedy"),
        final_extraction: String::from("tensat_ilp"),
        cost_threshold: 0.1,
        iter_limit: 200,
        prune_actions: true,
        rollout_strategy: String::from("random"),
        subtree_caching: false,
        select_max_uct_action: true,

        // experiment tracking
        seed: 0,
        experiments_base_path: String::from("/usr/experiments/tensor_eqs_mcts/"),
    };

    // // Experiment: extraction methods (egg_greedy, new_greedy, tensat_ilp)
    let seed = args
        .value_of("seed")
        .unwrap()
        .parse::<usize>()
        .unwrap();
    settings.seed = seed;

    let model = args
        .value_of("model")
        .unwrap()
        .parse::<String>()
        .unwrap();
    settings.model = model.clone();

    let extraction = args
        .value_of("extraction")
        .unwrap()
        .parse::<String>()
        .unwrap();
    settings.extraction = extraction.clone();

    let final_extraction = args
        .value_of("final_extraction")
        .unwrap()
        .parse::<String>()
        .unwrap();
    settings.final_extraction = final_extraction.clone();

    settings.experiments_base_path = String::from("/usr/experiments/tensor_eqs_mcts/") + &extraction + "_" + &final_extraction + "/";

    if model == String::from("nasrnn") {
        settings.multi_rules = String::from("/usr/tensat/converted_multi_nasrnn.txt");
    } else {
        settings.multi_rules = String::from("/usr/tensat/converted_multi.txt");
    }
    
    optimize(settings);

    // test_extractor();
}

fn test_extractor() {
    let model = String::from("138_model");

    let expr = match model.as_str() {
        "resnet50" => resnet50::get_resnet50(),
        "nasrnn" => nasrnn::get_nasrnn(),
        "resnext50" => resnext50::get_resnext50(),
        "bert" => bert::get_bert(),
        "nasneta" => nasneta::get_nasneta(),
        "inceptionv3" => inceptionv3::get_inceptionv3(),
        "mobilenetv2" => mobilenetv2::get_mobilenetv2(),
        "vgg" => vgg::get_vgg(),
        "squeezenet" => squeezenet::get_squeezenet(),
        "simple_example" => get_simple_model(),
        "complex_example" => get_complex_model(),
        "complex_example2" => get_complex_model2(),
        "suboptimal" => get_suboptimal_model(),
        "138_model" => get_138_model(),
        _ => panic!("The model name is not supported"),
    };

    let mut runner: Runner<Mdl, TensorAnalysis> = Runner::default().with_expr(&expr);
    let root = runner.roots[0];

    assert!(&runner.egraph.total_size() == &runner.egraph.number_of_classes());

    // save_model(&runner, "/usr/rmcts/suboptimal_model.model");
    // &runner.egraph.dot().to_svg("/usr/rmcts/suboptimal_model.svg").unwrap();

    let cost_model: CostModel = tensat::optimize::CostModel::with_setting(true);
    let mut tnsr_cost: TensorCost = tensat::optimize::TensorCost::new(&runner.egraph, &cost_model, true);

    // TENSAT ILP
    let ilp_dir = Path::new("/usr/experiments/rmcts/tests/").join("ilp").into_os_string().into_string().unwrap();
    let (expr, cost, duration) = extract_by_ilp_rmcts(&runner.egraph, root, &cost_model, false, false, true, false, 600, 1, ilp_dir, false);
    println!("TENSAT ILP base cost: {}, duration: {}", cost, duration);

    // Egg ILP
    let now = Instant::now();
    let (base_cost, best_expr) = LpExtractor::new(&runner.egraph, tnsr_cost.clone()).solve(root);
    println!("Egg ILP base cost: {}, duration: {}", base_cost, now.elapsed().as_millis());

    // Default greedy
    let now = Instant::now();
    let extractor = Extractor::new(&runner.egraph, tnsr_cost);
    let (base_cost, best_expr) = extractor.find_best(root);
    println!("Egg greedy base cost: {}, duration: {}", base_cost, now.elapsed().as_millis());

    // Fixed greedy
    let mut tnsr_cost: TensorCost = tensat::optimize::TensorCost::new(&runner.egraph, &cost_model, false);
    let now = Instant::now();
    let extractor = Extractor::new(&runner.egraph, tnsr_cost);
    let (base_cost, best_expr) = extractor.find_best(root);
    println!("New greedy base cost: {}, duration: {}", base_cost, now.elapsed().as_millis());
}

fn optimize(settings: Settings) {
    let filter_after = !settings.filter_before; // filter cycles after applying rewrite rules

    let output_dir = Path::new(&settings.experiments_base_path).join(settings.model.clone() + "_" + &settings.seed.to_string());

    let expr = match settings.model.as_str() {
        "resnet50" => resnet50::get_resnet50(),
        "nasrnn" => nasrnn::get_nasrnn(),
        "resnext50" => resnext50::get_resnext50(),
        "bert" => bert::get_bert(),
        "nasneta" => nasneta::get_nasneta(),
        "inceptionv3" => inceptionv3::get_inceptionv3(),
        "mobilenetv2" => mobilenetv2::get_mobilenetv2(),
        "vgg" => vgg::get_vgg(),
        "squeezenet" => squeezenet::get_squeezenet(),
        "very_simple_example" => get_very_simple_model(),
        "simple_example" => get_simple_model(),
        "complex_example" => get_complex_model(),
        "complex_example2" => get_complex_model2(),
        "suboptimal" => get_suboptimal_model(),
        "138_model" => get_138_model(),
        _ => panic!("The model name is not supported"),
    };

    // Load single learned and pre-defined TASO rules
    let learned_rules = read_to_string(settings.rules.clone()).expect("Something went wrong reading the rule file");
    let pre_defined_rules = tensat::rewrites::PRE_DEFINED_RULES.iter().map(|&x| x);
    let split_rules: Vec<&str> = learned_rules.split("\n").chain(pre_defined_rules).collect();
    let do_filter_after = settings.no_cycle && filter_after;
    let rules = tensat::rewrites::rules_from_str(split_rules, do_filter_after);
    println!("#Single-pattern rewrite rules: {}", rules.len());

    // Load multi learned and pre-defined TASO rules
    let multi_patterns = if settings.use_multi {
        let learned_multi_rules = read_to_string(settings.multi_rules.clone()).expect("Something went wrong reading the multi rule file");
        let pre_defined_multi_rules = tensat::rewrites::PRE_DEFINED_MULTI.iter().map(|&x| (x, false));
        let multi_rules: Vec<(&str, bool)> = learned_multi_rules.split("\n").map(|x| (x, true)).chain(pre_defined_multi_rules).collect();
        let multi_patterns = MultiPatterns::with_rules(multi_rules.clone(), settings.no_cycle, settings.iter_multi, filter_after, settings.node_multi, settings.n_sec, String::new());
        println!("#Multi-pattern rewritue rules: {}", multi_patterns.rules.len());
        Some(multi_patterns)
    } else {
        println!("Multi-pattern rewrite rules will not be used!");
        None
    };

    // RMCTS args
    let n_threads = std::thread::available_parallelism().unwrap().get();
    let args = MCTSArgs {
        // mcts
        budget: settings.budget,
        max_sim_step: settings.max_sim_step,
        gamma: settings.gamma,
        expansion_worker_num: settings.expansion_worker_num,
        simulation_worker_num: settings.simulation_worker_num,
        all_weight_only: settings.all_weight_only,
        extraction: settings.extraction.clone(),
        final_extraction: settings.final_extraction.clone(),
        cost_threshold: settings.cost_threshold,
        iter_limit: settings.iter_limit,
        prune_actions: settings.prune_actions,
        rollout_strategy: settings.rollout_strategy.clone(),
        subtree_caching: settings.subtree_caching,
        select_max_uct_action: settings.select_max_uct_action,
        // egg
        node_limit: settings.n_nodes,
        time_limit: settings.n_sec as usize,
        // experiment tracking
        output_dir: output_dir.clone(),
        save_graph: settings.save_graph.clone(),
        export_models: settings.export_models,
        // tensat ilp
        order_var_int: settings.order_var_int,
        class_constraint: settings.class_constraint,
        no_order: settings.no_order,
        initial_with_greedy: settings.initial_with_greedy,
        ilp_time_sec: settings.ilp_time_sec,
        ilp_num_threads: settings.ilp_num_threads,
    };

    // Warn if output directory already exists, otherwise create it
    if output_dir.exists() {
        println!("You are overwriting existing data!");
    } else {
        create_dir_all(&output_dir);
    }

    // Save settings to file
    save_data_to_file(&settings, &output_dir, "settings.txt");

    // Init runner to avoid CUDNN/CUDA errors and to get egraph and root id
    let runner = Runner::<Mdl, TensorAnalysis, ()>::default().with_expr(&expr);
    let root = runner.roots[0];

    run::run_mcts(runner.egraph, root, rules, multi_patterns, Some(args));
}