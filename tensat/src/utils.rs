#![allow(unused)]
// source: tensat/src/main.rs

use clap::{App, Arg};
use egg::*;
use std::collections::{HashMap, HashSet};
use std::env::*;
use std::fs::*;
use std::time::*;
use std::time::{Duration, Instant};
use crate::model::*;
use crate::optimize::*;
use crate::rewrites::*;
use crate::{parse::*, verify::*};

use serde::{Deserialize, Serialize};
use serde_json::json;
use std::fs::OpenOptions;
use std::io::prelude::*;
use std::io::Error;
use std::process::{Command, Stdio};
use std::thread;
use std::path::Path;
use std::ffi::CString;

pub fn save_model(runner: &Runner<Mdl, TensorAnalysis, ()>, file_name: &str) {
    // let mut g = runner.egraph.analysis.graph.borrow_mut();
    let mut g = runner.egraph.analysis.graph.lock().unwrap();
    unsafe {
        (*g).export_to_file_raw(CString::new(file_name).unwrap().into_raw());
    }
}

pub fn get_full_graph_runtime(runner: &Runner<Mdl, TensorAnalysis, ()>, process: bool) -> f32 {
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


/// Extract the optimal graph from EGraph by ILP
///
/// This function prepares the data for the ILP formulation, save it as json, call the python
/// script to read the data + solve ILP + save the solved results. After the python script
/// finishes, it reads back the solved result and construct the RecExpr for the optimized graph.
pub fn extract_by_ilp_rmcts(
    egraph: &EGraph<Mdl, TensorAnalysis>,
    root: Id,
    cost_model: &CostModel,
    order_var_int: bool,
    class_constraint: bool,
    no_order: bool,
    initialize: bool,
    ilp_time_sec: usize,
    ilp_num_threads: usize,
    ilp_dir: String,
    return_expr: bool,
) -> (RecExpr<Mdl>, f32, f32) {
    let binding = std::thread::current();
    let thread_name = binding.name().unwrap();
    println!("Thread {} went into extract_by_ilp_rmcts.", thread_name);
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
    create_dir_all(ilp_dir.clone());
    let filename = Path::new(&ilp_dir).join("ilp_data_".to_owned() + thread_name + ".json");
    write(filename, data_str).expect("Unable to write file");

    // let initialize = matches.is_present("initial_with_greedy");
    if initialize {
        // Get node_to_i map
        let node_to_i: HashMap<Mdl, usize> = (&i_to_nodes)
            .iter()
            .enumerate()
            .map(|(i, node)| (node.clone(), i))
            .collect();

        // let tnsr_cost = TensorCost {
        //     egraph,
        //     cost_model,
        // };
        let tnsr_cost = TensorCost::new(egraph, cost_model, false); // use fixed greedy extractor
        
        let extractor = Extractor::new(egraph, tnsr_cost);
        let (i_list, m_list) = get_init_solution(egraph, root, &extractor, &g_i, &node_to_i);

        // Store initial solution
        let solution_data = json!({
            "i_list": i_list,
            "m_list": m_list,
        });
        let sol_data_str = serde_json::to_string(&solution_data).expect("Fail to convert json to string");
        let filename = Path::new(&ilp_dir).join("init_sol_".to_owned() + thread_name + ".json");
        write(filename, sol_data_str).expect("Unable to write file");
    }

    // Call python script to run ILP
    // let order_var_int = matches.is_present("order_var_int");
    // let class_constraint = matches.is_present("class_constraint");
    // let no_order = false matches.is_present("no_order");

    let mut arg_vec = vec!["/usr/tensat/extractor/extract.py"];
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

    let binding = ilp_time_sec.to_string();
    arg_vec.push("--time_lim_sec");
    arg_vec.push(&binding);

    arg_vec.push("--num_thread");
    let binding = ilp_num_threads.to_string();
    arg_vec.push(&binding);

    arg_vec.push("--output_dir");
    arg_vec.push(&ilp_dir);

    arg_vec.push("--thread_name");
    arg_vec.push(thread_name);

    if false {
        arg_vec.push("--verbose");
    }

    // if let Some(time_lim) = matches.value_of("ilp_time_sec") {
    //     arg_vec.push("--time_lim_sec");
    //     arg_vec.push(time_lim);
    // }

    // if let Some(num_thread) = matches.value_of("ilp_num_threads") {
    //     arg_vec.push("--num_thread");
    //     arg_vec.push(num_thread);
    // }

    let child = Command::new("python")
        .args(&arg_vec)
        .spawn()
        .expect("failed to execute child");
    let output = child.wait_with_output().expect("failed to get output");

    if output.status.success() {
        // Read back solved results, construct optimized graph
        let filename = Path::new(&ilp_dir).join("solved_".to_owned() + thread_name + ".json");

        let solved_str = read_to_string(filename)
            .expect("Something went wrong reading the solved file");

        let solved_data: SolvedResults =
            serde_json::from_str(&solved_str).expect("JSON was not well-formatted");

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
        if return_expr {
            // println!("Egg expression builder started.");
            let root_enode = node_picked.get(&root).unwrap();
            expr = root_enode.build_recexpr(|child| node_picked.get(&child).unwrap().clone());
            // println!("Egg expression builder finished.");
        };
        
        // let mut tensat_expr = RecExpr::default();
        // let mut added_memo: HashMap<Id, Id> = Default::default();
        // println!("TENSAT expression builder started.");
        // Can lead to a stackoverflow
        // let _ = construct_best_rec(&node_picked, root, &mut added_memo, egraph, &mut tensat_expr);
        // println!("TENSAT expression builder finished.");

        (expr, solved_data.cost, solved_data.time)
    } else {
        println!("Output status: {}", output.status);
        panic!("Python script failed");
    }
}