#![allow(unused_variables)]
#![allow(unused_must_use)]

use clap::{App, Arg};
use egg::*;
use std::collections::{HashMap};
use std::fs::*;
use std::time::{Duration, Instant};
use tensat::bert;
use tensat::model::*;
use tensat::nasneta;
use tensat::nasrnn;
use tensat::optimize::*;
use tensat::resnet50;
use tensat::resnext50;
use tensat::rewrites::*;
use tensat::inceptionv3;
use tensat::mobilenetv2;
use tensat::vgg;
use tensat::squeezenet;
use tensat::{parse::*, verify::*};

use serde_json::{json, Map, Value};
use std::fs::OpenOptions;
use std::io::prelude::*;
use std::process::{Command};
use std::path::Path;

use std::ffi::CString;

fn main() {
    // Parse arguments
    let matches = App::new("Tamago")
        .arg(
            Arg::with_name("mode")
                .short("m")
                .long("mode")
                .takes_value(true)
                .default_value("optimize")
                .help("Mode to run, can be verify, optimize, test, convert"),
        )
        .arg(
            Arg::with_name("model")
                .short("d")
                .long("model")
                .takes_value(true)
                .help("Specify a pre-defined model to optimize"),
        )
        .arg(
            Arg::with_name("rules")
                .short("r")
                .long("rules")
                .takes_value(true)
                .help("Provide a file with rewrite rules"),
        )
        .arg(
            Arg::with_name("gj")
                .long("gj"),
        )
        .arg(
            Arg::with_name("out_file")
                // .short("o")
                .long("out_file")
                .takes_value(true)
                .help("Provide a output file name. For mode convert, it's for converted rules; for mode optimize, it's for measured runtime"),
        )
        .arg(
            Arg::with_name("export_models")
                .short("x")
                .long("export_models")
                .help("Whether or not to store input and optimized model"),
        )
        .arg(
            Arg::with_name("model_file")
                .short("f")
                .long("model_file")
                .takes_value(true)
                .help("Provide a file with the input model"),
        )
        .arg(
            Arg::with_name("multi_rules")
                .short("t")
                .long("multi_rules")
                .takes_value(true)
                .help("File with multi-pattern rules. Every two lines belong to one multi-pattern rule"),
        )
        .arg(
            Arg::with_name("save_graph")
                .short("s")
                .long("save_graph")
                .takes_value(true)
                .default_value("io")
                .help("Whether to save graphs as dot files. Can be: all, io, none"),
        )
        .arg(
            Arg::with_name("use_multi")
                .short("u")
                .long("use_multi")
                .help("Set this flag will enable use of multi-pattern rules"),
        )
        .arg(
            Arg::with_name("n_iter")
                .long("n_iter")
                .takes_value(true)
                .default_value("3")
                .help("Max number of iterations for egg to run"),
        )
        .arg(
            Arg::with_name("n_sec")
                .long("n_sec")
                .takes_value(true)
                .default_value("10")
                .help("Max number of seconds for egg to run"),
        )
        .arg(
            Arg::with_name("n_nodes")
                .long("n_nodes")
                .takes_value(true)
                .default_value("100000")
                .help("Max number of nodes for egraph"),
        )
        .arg(
            Arg::with_name("extract")
                .short("e")
                .long("extract")
                .takes_value(true)
                .default_value("greedy")
                .help("Extraction method, can be greedy, ilp"),
        )
        .arg(
            Arg::with_name("order_var_int")
                .long("order_var_int")
                .help("Set this flag will let ILP use integer var for ordering"),
        )
        .arg(
            Arg::with_name("class_constraint")
                .long("class_constraint")
                .help("Add constraint in ILP that each eclass sum to 1"),
        )
        .arg(
            Arg::with_name("no_order")
                .long("no_order")
                .help("No ordering constraints in ILP"),
        )
        .arg(
            Arg::with_name("initial_with_greedy")
                .long("initial_with_greedy")
                .help("Initialize ILP with greedy solution"),
        )
        .arg(
            Arg::with_name("ilp_time_sec")
                .long("ilp_time_sec")
                .takes_value(true)
                .help("Time limit for ILP solver (seconds)"),
        )
        .arg(
            Arg::with_name("ilp_num_threads")
                .long("ilp_num_threads")
                .takes_value(true)
                .help("Number of threads for ILP solver"),
        )
        .arg(
            Arg::with_name("iter_multi")
                .long("iter_multi")
                .takes_value(true)
                .default_value("1")
                .help("Max number of iterations to apply multi-pattern rules"),
        )
        .arg(
            Arg::with_name("node_multi")
                .long("node_multi")
                .takes_value(true)
                .default_value("3000000")
                .help("Max number of nodes added by multi-pattern rules"),
        )
        .arg(
            Arg::with_name("no_cycle")
                .long("no_cycle")
                .help("Not allowing cycles in EGraph"),
        )
        .arg(
            Arg::with_name("filter_before")
                .long("filter_before")
                .help("Filter cycles before applying rules"),
        )
        .arg(
            Arg::with_name("all_weight_only")
                .long("all_weight_only")
                .help("Treat zero cost for all weight concat only"),
        )
        .arg(
            Arg::with_name("saturation_only")
                .long("saturation_only")
                .help("Run saturation only"),
        )
        .arg(
            Arg::with_name("output_dir")
                .long("output_dir")
                .short("out_dir")
                .takes_value(true)
                .help("Output directory to save all experimental results to"),
        )
        .get_matches();

    let run_mode = matches.value_of("mode").unwrap();
    println!("Running mode is: {}", run_mode);

    match run_mode {
        "optimize" => optimize(matches),
        "verify" => prove_taso_rules(matches),
        "test" => test(matches),
        "convert" => convert_learned_rules(matches),
        _ => panic!("Running mode not supported"),
    }
}

fn convert_learned_rules(matches: clap::ArgMatches) {
    env_logger::init();

    let file = matches
        .value_of("rules")
        .expect("Pls supply taso rules file.");
    let outf = matches.value_of("out_file").unwrap_or("converted.txt");
    let taso_rules = read_to_string(file).expect("Something went wrong reading the file");

    let converted = parse_and_convert(&taso_rules);

    write(outf, converted).expect("Unable to write file");
}

fn test(matches: clap::ArgMatches) {}

/// Main procedure to run optimization
///
/// Gets input graph and rewrite rules; runs saturation with TensorAnalysis dealing with metadata; runs
/// greedy extraction with TensorCost getting the cost per node/op; evaluates
/// full graph runtime of the starting graph and extracted graph.
fn optimize(matches: clap::ArgMatches) {
    env_logger::init();

    // Read settings from args
    let rule_file = matches
        .value_of("rules")
        .expect("Pls supply rewrite rules file.");
    let save_graph = matches.value_of("save_graph").unwrap();
    let use_multi = matches.is_present("use_multi");
    let no_cycle = matches.is_present("no_cycle");
    let filter_after = !matches.is_present("filter_before");
    let output_directory = matches.value_of("output_dir").unwrap();

    // Warn if output directory already exists, otherwise create it
    if Path::new(output_directory).exists() {
        println!("You are overwriting existing data!");
    } else {
        create_dir_all(output_directory);
    }

    // Save settings
    let mut settings = Map::new();
    for arg in &matches.args {
        let key = arg.0.to_string();
        let value = if !arg.1.vals.is_empty() {
            Value::String(arg.1.vals[0].clone().into_string().unwrap())
        } else {
            Value::String(String::from("true"))
        };
        settings.insert(key, value);
    }

    let filename = Path::new(output_directory).join("settings.txt");
    let mut file = OpenOptions::new().append(true).create(true).open(filename).unwrap();
    let settings_data = serde_json::to_string(&settings).expect("Failed to convert json to string");

    if let Err(e) = writeln!(file, "{}", settings_data) {
        eprintln!("Couldn't write to file: {}", e);
    }

    // Get input graph and rules
    // learned_rules are the learned rules from TASO, pre_defined_rules are the hand-specified rules from TASO
    let learned_rules =
        read_to_string(rule_file).expect("Something went wrong reading the rule file");
    let pre_defined_rules = PRE_DEFINED_RULES.iter().map(|&x| x);
    let split_rules: Vec<&str> = learned_rules.split("\n").chain(pre_defined_rules).collect();
    let do_filter_after = no_cycle && filter_after;
    let rules = rules_from_str(split_rules, do_filter_after);

    let start = match matches.value_of("model") {
        Some("resnet50") => resnet50::get_resnet50(),
        Some("nasrnn") => nasrnn::get_nasrnn(),
        Some("resnext50") => resnext50::get_resnext50(),
        Some("bert") => bert::get_bert(),
        Some("nasneta") => nasneta::get_nasneta(),
        Some("inceptionv3") => inceptionv3::get_inceptionv3(),
        Some("mobilenetv2") => mobilenetv2::get_mobilenetv2(),
        Some("vgg") => vgg::get_vgg(),
        Some("squeezenet") => squeezenet::get_squeezenet(),
        Some(_) => panic!("The model name is not supported"),
        None => {
            let model_file = matches
                .value_of("model_file")
                .expect("Pls supply input graph file.");
            let input_graph =
                read_to_string(model_file).expect("Something went wrong reading the model file");
            input_graph.parse().unwrap()
        }
    };

    // Get multi-pattern rules. learned_rules are the learned rules from TASO,
    // pre_defined_multi are the hand-specified rules from TASO
    let n_sec = matches.value_of("n_sec").unwrap().parse::<u64>().unwrap();
    let iter_multi = matches
        .value_of("iter_multi")
        .unwrap()
        .parse::<usize>()
        .unwrap();
    let node_multi = matches
        .value_of("node_multi")
        .unwrap()
        .parse::<usize>()
        .unwrap();
    let mut multi_patterns = if let Some(rule_file) = matches.value_of("multi_rules") {
        let learned_rules =
            read_to_string(rule_file).expect("Something went wrong reading the rule file");
        let pre_defined_multi = PRE_DEFINED_MULTI.iter().map(|&x| (x, /*symmetric=*/ false));
        // The learned rules we have are symmetric. Predefined ones are not
        let multi_rules: Vec<(&str, bool)> = learned_rules
            .split("\n")
            .map(|x| (x, /*symmetric=*/ true))
            .chain(pre_defined_multi)
            .collect();
        MultiPatterns::with_rules(multi_rules, no_cycle, iter_multi, filter_after, node_multi, n_sec, String::from(output_directory))
    } else {
        let multi_rules: Vec<(&str, bool)> = PRE_DEFINED_MULTI
            .iter()
            .map(|&x| (x, /*symmetric=*/ false))
            .collect();
        MultiPatterns::with_rules(multi_rules, no_cycle, iter_multi, filter_after, node_multi, n_sec, String::from(output_directory))
    };

    // Run saturation
    let time_limit_sec = Duration::new(n_sec, 0);
    let iter_limit = matches
        .value_of("n_iter")
        .unwrap()
        .parse::<usize>()
        .unwrap();
    let node_limit = matches
        .value_of("n_nodes")
        .unwrap()
        .parse::<usize>()
        .unwrap();

    let runner = if use_multi {
        // This hook function (which applies the multi-pattern rules) will be called at the
        // beginning of each iteration in equality saturation
        Runner::<Mdl, TensorAnalysis, ()>::default()
            .with_node_limit(node_limit)
            .with_time_limit(time_limit_sec)
            .with_iter_limit(iter_limit)
            .with_expr(&start)
            .with_hook(move |runner| multi_patterns.run_one(runner, None))
    } else {
        Runner::<Mdl, TensorAnalysis, ()>::default()
            .with_node_limit(node_limit)
            .with_time_limit(time_limit_sec)
            .with_iter_limit(iter_limit)
            .with_expr(&start)
    };

    // if matches.is_present("gj") {
    //     runner.egraph.strategy = egg::Strategy::GenericJoin;
    // } else {
    //     runner.egraph.strategy = egg::Strategy::EMatch;
    // }

    let start_time = Instant::now();
    let mut runner = runner.run(&rules[..]);

    if do_filter_after {
        // Do cycle removal after the final iteration
        remove_cycle_by_order(&mut runner);
    }
    let sat_duration = start_time.elapsed();
    let num_iter_sat = runner.iterations.len() - 1;

    println!("Runner complete!");
    println!("  Nodes: {}", runner.egraph.total_size());
    println!("  Classes: {}", runner.egraph.number_of_classes());
    println!("  Stopped: {:?}", runner.stop_reason.as_ref().unwrap());
    println!("  Time taken: {:?}", sat_duration);
    println!("  Number of iterations: {:?}", num_iter_sat);

    let (num_enodes, num_classes, avg_nodes_per_class, num_edges, num_programs) =
        get_stats(&runner.egraph);
    println!("  Average nodes per class: {}", avg_nodes_per_class);
    println!("  Number of edges: {}", num_edges);
    println!("  Number of programs: {}", num_programs);

    // Save iteration data
    let filename = Path::new(output_directory).join("iteration_data.txt");
    let mut file = OpenOptions::new().append(true).create(true).open(filename).unwrap();
    let iteration_data = serde_json::to_string(&runner.iterations).expect("Failed to convert IterationData json to string");
    if let Err(e) = writeln!(file, "{}", iteration_data) {
        eprintln!("Couldn't write to file: {}", e);
    }

    // Save egraph
    let (egraph, root) = (runner.egraph, runner.roots[0]);
    if save_graph == "all" {
        let filename = Path::new(output_directory).join("tensat.svg");
        egraph.dot().to_svg(filename).unwrap();
    }

    if matches.is_present("saturation_only") {
        if let Some(outf) = matches.value_of("out_file") {
            let filename = Path::new(output_directory).join(outf);
            let mut file = OpenOptions::new()
                .append(true)
                .create(true)
                .open(filename)
                .unwrap();

            // Stats to write: original runtime, optimized runtime, saturation time, extraction time,
            // number of nodes, number of eclasses, number of possible programs
            let data = json!({
                "runner_stop_reason": runner.stop_reason.as_ref().unwrap(),
                "runner_time": sat_duration.as_secs_f32(),
                "num_iterations": num_iter_sat,
                "num_enodes": num_enodes,
                "num_classes": num_classes,
                "avg_nodes_per_class": avg_nodes_per_class,
                "num_edges": num_edges,
                "num_programs": num_programs,
                "extraction_time": 0.0,
                "original_runtime": 0.0,
                "optimized_runtime": 0.0,
            });
            let sol_data_str = serde_json::to_string(&data).expect("Fail to convert json to string");

            if let Err(e) = writeln!(file, "{}", sol_data_str) {
                eprintln!("Couldn't write to file: {}", e);
            }
        }
    } else {
        // Run extraction
        let extract_mode = matches.value_of("extract").unwrap();
        let cost_model = CostModel::with_setting(
            /*ignore_all_weight_only=*/ matches.is_present("all_weight_only"),
        );
        let (best, best_cost, ext_secs) = match extract_mode {
            "ilp" => extract_by_ilp(&egraph, root, &matches, &cost_model),
            "egg_ilp" => {
                let tnsr_cost = TensorCost::new(
                    &egraph,
                    &cost_model,
                    true,
                );

                let start_time = Instant::now();
                let mut lp_extractor = LpExtractor::new(&egraph, tnsr_cost);
                let (best_cost, best) = lp_extractor.solve(root);
                let duration = start_time.elapsed();

                println!("Egg's ILP Extractor complete!");
                println!("  Time taken: {:?}", duration);
                println!("  Best cost: {:?}", best_cost);
                (best, best_cost as f32, duration.as_secs_f32())
            }
            "greedy" => {
                let tnsr_cost = TensorCost::new(
                    &egraph,
                    &cost_model,
                    true,
                );
                let start_time = Instant::now();
                let extractor = Extractor::new(&egraph, tnsr_cost);
                let (best_cost, best) = extractor.find_best(root);
                let duration = start_time.elapsed();

                println!("Extractor complete!");
                println!("  Time taken: {:?}", duration);
                println!("  Best cost: {:?}", best_cost);
                (best, best_cost, duration.as_secs_f32())
            }
            _ => panic!("Extracting mode not supported"),
        };

        // Evaluation starting and extracted graph runtime, save graphs
        let runner_start = Runner::<Mdl, TensorAnalysis, ()>::default().with_expr(&start);
        let runner_ext = Runner::<Mdl, TensorAnalysis, ()>::default().with_expr(&best);

        if save_graph != "none" {
            let start_filename = Path::new(output_directory).join("start.svg");
            runner_start.egraph.dot().to_svg(start_filename).unwrap();

            let ext_filename = Path::new(output_directory).join("ext.svg");
            runner_ext.egraph.dot().to_svg(ext_filename).unwrap();
        }

        let time_start = get_full_graph_runtime(&runner_start, false);
        println!("Start graph runtime: {}", time_start);

        let time_ext = get_full_graph_runtime(&runner_ext, true);
        println!("Extracted graph runtime: {}", time_ext);

        if matches.is_present("export_models") {
            let filename_start = Path::new(output_directory).join("start.model");
            save_model(&runner_start, filename_start.to_str().unwrap());

            let filename_optimized = Path::new(output_directory).join("optimized.model");
            save_model(&runner_ext, filename_optimized.to_str().unwrap());
        }

        if let Some(outf) = matches.value_of("out_file") {
            let filename = Path::new(output_directory).join(outf);
            let mut file = OpenOptions::new()
                .append(true)
                .create(true)
                .open(filename)
                .unwrap();

            // Stats to write: original runtime, optimized runtime, saturation time, extraction time,
            // number of nodes, number of eclasses, number of possible programs
            let data = json!({
                "runner_stop_reason": runner.stop_reason.as_ref().unwrap(),
                "runner_time": sat_duration.as_secs_f32(),
                "num_iterations": num_iter_sat,
                "num_enodes": num_enodes,
                "num_classes": num_classes,
                "avg_nodes_per_class": avg_nodes_per_class,
                "num_edges": num_edges,
                "num_programs": num_programs,
                "extraction_time": ext_secs,
                "original_runtime": time_start,
                "optimized_runtime": time_ext,
            });
            let sol_data_str = serde_json::to_string(&data).expect("Fail to convert json to string");

            if let Err(e) = writeln!(file, "{}", sol_data_str) {
                eprintln!("Couldn't write to file: {}", e);
            }
        }
    }
}

/// Extract the optimal graph from EGraph by ILP
///
/// This function prepares the data for the ILP formulation, save it as json, call the python
/// script to read the data + solve ILP + save the solved results. After the python script
/// finishes, it reads back the solved result and construct the RecExpr for the optimized graph.
fn extract_by_ilp(
    egraph: &EGraph<Mdl, TensorAnalysis>,
    root: Id,
    matches: &clap::ArgMatches,
    cost_model: &CostModel,
) -> (RecExpr<Mdl>, f32, f32) {
    let binding = std::thread::current();
    let thread_name = binding.name().unwrap();
    // Prepare data for ILP formulation, save to json
    let (m_id_map, e_m, h_i, cost_i, g_i, root_m, i_to_nodes, blacklist_i) =
        prep_ilp_data(egraph, root, cost_model);

    let data = json!({
        "e_m": e_m,
        "h_i": h_i,
        "cost_i": cost_i,
        "g_i": g_i,
        "root_m": root_m,
        "blacklist_i": blacklist_i,
    });
    let data_str = serde_json::to_string(&data).expect("Fail to convert json to string");

    let output_directory = matches.value_of("output_dir").unwrap();
    let filename = Path::new(output_directory).join("ilp_data_".to_owned() + thread_name + ".json");
    write(filename, data_str).expect("Unable to write file");

    let initialize = matches.is_present("initial_with_greedy");
    if initialize {
        // Get node_to_i map
        let node_to_i: HashMap<Mdl, usize> = (&i_to_nodes)
            .iter()
            .enumerate()
            .map(|(i, node)| (node.clone(), i))
            .collect();

        let tnsr_cost = TensorCost::new(
            egraph,
            cost_model,
            true,
        );
        let extractor = Extractor::new(egraph, tnsr_cost);
        let (i_list, m_list) = get_init_solution(egraph, root, &extractor, &g_i, &node_to_i);

        // Store initial solution
        let solution_data = json!({
            "i_list": i_list,
            "m_list": m_list,
        });
        let sol_data_str = serde_json::to_string(&solution_data).expect("Fail to convert json to string");
        let filename = Path::new(output_directory).join("init_sol_".to_owned() + thread_name + ".json");
        write(filename, sol_data_str).expect("Unable to write file");
    }

    // Call python script to run ILP
    let order_var_int = matches.is_present("order_var_int");
    let class_constraint = matches.is_present("class_constraint");
    let no_order = matches.is_present("no_order");
    let mut arg_vec = vec!["extractor/extract.py"];
    if order_var_int {
        arg_vec.push("--order_var_int");
    }
    if class_constraint {
        arg_vec.push("--eclass_constraint");
    }
    if no_order {
        arg_vec.push("--no_order");
    }
    if initialize {
        arg_vec.push("--initialize")
    }
    if let Some(time_lim) = matches.value_of("ilp_time_sec") {
        arg_vec.push("--time_lim_sec");
        arg_vec.push(time_lim);
    }
    if let Some(num_thread) = matches.value_of("ilp_num_threads") {
        arg_vec.push("--num_thread");
        arg_vec.push(num_thread);
    }
    if let Some(output_dir) = matches.value_of("output_dir") {
        arg_vec.push("--output_dir");
        arg_vec.push(output_dir);
    }
    arg_vec.push("--thread_name");
    arg_vec.push(thread_name);
    let child = Command::new("python")
        .args(&arg_vec)
        .spawn()
        .expect("failed to execute child");
    let output = child.wait_with_output().expect("failed to get output");

    if output.status.success() {
        // Read back solved results, construct optimized graph
        let filename = Path::new(output_directory).join("solved_".to_owned() + thread_name + ".json");
        let solved_str = read_to_string(filename).expect("Something went wrong reading the solved file");
        let solved_data: SolvedResults = serde_json::from_str(&solved_str).expect("JSON was not well-formatted");

        let mut node_picked: HashMap<Id, Mdl> = HashMap::new();
        for (i, x_i) in solved_data.solved_x.iter().enumerate() {
            if *x_i == 1 {
                let eclass_id = m_id_map[g_i[i]];
                if node_picked.contains_key(&eclass_id) {
                    println!("Duplicate node in eclass");
                    println!("{}", node_picked.get(&eclass_id).unwrap());
                    println!("{}", i_to_nodes[i]);
                    continue;
                }
                //assert!(!node_picked.contains_key(&eclass_id));
                node_picked.insert(eclass_id, i_to_nodes[i].clone());
            }
        }

        let mut expr = RecExpr::default();
        let mut added_memo: HashMap<Id, Id> = Default::default();
        let _ = construct_best_rec(&node_picked, root, &mut added_memo, egraph, &mut expr);
        (expr, solved_data.time, solved_data.time)
    } else {
        panic!("Python script failed");
    }
}

/// This function gets the following stats:
///     Total number of enodes
///     Total number of eclasses
///     Average number of enodes per class
///     Total number of edges (children relationships)
///     Total number of equivalent programs represented (power of 2)
fn get_stats(egraph: &EGraph<Mdl, TensorAnalysis>) -> (usize, usize, f32, usize, f32) {
    let num_enodes = egraph.total_size();
    let num_classes = egraph.number_of_classes();
    let avg_nodes_per_class = num_enodes as f32 / (num_classes as f32);
    let num_edges = egraph
        .classes()
        .fold(0, |acc, c| c.iter().fold(0, |sum, n| n.len() + sum) + acc);
    let num_programs = egraph
        .classes()
        .fold(0.0, |acc, c| acc + (c.len() as f32).log2());
    (
        num_enodes,
        num_classes,
        avg_nodes_per_class,
        num_edges,
        num_programs,
    )
}

fn get_full_graph_runtime(runner: &Runner<Mdl, TensorAnalysis, ()>, process: bool) -> f32 {
    // let mut g = runner.egraph.analysis.graph.borrow_mut();
    let mut g = runner.egraph.analysis.graph.lock().unwrap();
    unsafe {
        // This is calling TASO's preprocess_weights function before evaluating full graph
        // run time. It removes op that has only weights as its inputs. Since TASO only cares
        // about inference time, such ops can be pre-computed
        if process {
            let processed_g = g.preprocess_weights();
            // (*processed_g).export_to_file_raw(CString::new("/usr/tensat/optimized.onnx").unwrap().into_raw());
            (*processed_g).run()
        } else {
            //(*g).export_to_file_raw(CString::new("/usr/tensat/orig.onnx").unwrap().into_raw());
            (*g).run()
        }
    }
}

fn save_model(runner: &Runner<Mdl, TensorAnalysis, ()>, file_name: &str) {
    // let mut g = runner.egraph.analysis.graph.borrow_mut();
    let mut g = runner.egraph.analysis.graph.lock().unwrap();
    unsafe {
        (*g).export_to_file_raw(CString::new(file_name).unwrap().into_raw());
    }
}

fn prove_taso_rules(matches: clap::ArgMatches) {
    env_logger::init();

    let file = matches
        .value_of("rules")
        .expect("Pls supply taso rules file.");
    let taso_rules = read_to_string(file).expect("Something went wrong reading the file");

    println!("Parsing rules...");
    let initial = parse_rules(&taso_rules);
    println!("Parsed rules!");

    let mut to_prove = initial.clone();
    while !to_prove.is_empty() {
        let n_before = to_prove.len();
        to_prove = verify(&to_prove);
        let n_proved = n_before - to_prove.len();
        println!("Proved {} on this trip", n_proved);
        if n_proved == 0 {
            println!("\nCouldn't prove {} rule(s)", to_prove.len());
            for pair in &to_prove {
                let i = initial.iter().position(|p| p == pair).unwrap();
                println!("  {}: {} => {}", i, pair.0, pair.1);
            }
            break;
        }
    }
}
