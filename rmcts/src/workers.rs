use crate::eg_env::{Ckpt, EgraphEnv};
// use crate::env::Env;
use crate::tree::{ExpTask, SimTask};

#[allow(unused_imports)]
use egg::{
    Analysis, CostFunction, EGraph, Id, Language, LpCostFunction, RecExpr, Rewrite, StopReason,
};
use tensat::model::{Mdl, TensorAnalysis};
use tensat::rewrites::MultiPatterns;
use rand::{Rng};
use rand::distributions::{WeightedIndex, Distribution};
use std::sync::mpsc;
use std::thread;
use std::time::Duration;
use std::path::PathBuf;
use itertools::izip;

pub enum Message {
    Exit,
    #[allow(unused_variables)]
    Nothing,
    Expansion(ExpTask, u32, u32),
    Simulation(SimTask, u32),
}

pub enum Reply {
    OK,
    DoneExpansion(usize, (), f32, bool, bool, Vec<bool>, usize, Option<Ckpt>, u32, u32),
    DoneSimulation(u32, f32),
}

pub fn worker_loop(
    name: &'static str,
    id: usize,
    gamma: f32,
    max_sim_step: u32,
    verbose: bool,
    egraph: EGraph<Mdl, TensorAnalysis>,
    root_id: Id,
    rules: Vec<Rewrite<Mdl, TensorAnalysis>>,
    multi_patterns: Option<MultiPatterns>,
    all_weight_only: bool,
    extraction: String,
    prune_actions: bool,
    rollout_strategy: String,
    // egg
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
) -> (
    thread::JoinHandle<()>,
    mpsc::Sender<Message>,
    mpsc::Receiver<Reply>,
)
{
    let (tx, rx) = mpsc::channel();
    let (tx2, rx2) = mpsc::channel();
    let thread_identifier = name.to_owned() + "_" + id.to_string().as_str();
    let builder = thread::Builder::new().name(thread_identifier.into());
    let handle = builder.spawn(move || {
        // make env
        // let mut env = Env::new(expr, rules, node_limit, time_limit);
        let mut env = EgraphEnv::new(
            egraph, root_id, rules, multi_patterns, all_weight_only, extraction.clone(), prune_actions, node_limit, time_limit, order_var_int, class_constraint, no_order, initial_with_greedy, ilp_time_sec, ilp_num_threads, output_dir.clone()
        );
        // NOTE: Should this be reset at some point?
        let mut rewards = vec![0.0; env.get_action_space()];
        let mut visit_counts = vec![0.0; env.get_action_space()];
        // Get action space
        let action_n = env.get_action_space();
        env.reset();
        // worker loop
        loop {
            let message = rx.recv().unwrap();
            match message {
                Message::Exit => {
                    // println!("Worker {} Exit!", id);
                    break;
                }

                Message::Expansion(exp_task, global_saving_idx, task_idx) => {
                    if verbose {
                        println!("Worker {} Expansion!", id);
                    }
                    // expand one step
                    env.restore(exp_task.checkpoint_data);
                    let expand_action = exp_task.shallow_copy_node.select_expansion_action();
                    let (next_state, reward, done, info) = env.step(expand_action);

                    // saturated means this action doesn't match any enode
                    // so we shouldn't select again!
                    let mut child_saturated = false;
                    match info.report.stop_reason {
                        StopReason::Saturated => {
                            child_saturated = true;
                        }
                        _ => (),
                    }

                    // If the child is not saturated, we have to perform action pruning for the new child node
                    let children_saturated;
                    let children_saturated_cnt;
                    if !child_saturated {
                        (children_saturated, children_saturated_cnt) = env.action_pruning(vec![false; action_n]);
                    } else {
                        children_saturated = vec![true; action_n];
                        children_saturated_cnt = action_n;
                    }

                    // Checkpoint has to be done after action pruning to ensure that the saturation counter of the environment is correct
                    let new_checkpoint_data = if done { None } else { Some(env.checkpoint()) };

                    // reply
                    tx2.send(Reply::DoneExpansion(
                        expand_action,
                        next_state,
                        reward,
                        done,
                        child_saturated,
                        children_saturated,
                        children_saturated_cnt,
                        new_checkpoint_data,
                        global_saving_idx,
                        task_idx,
                    ))
                    .unwrap();
                }

                Message::Simulation(sim_task, task_idx) => {
                    env.restore(sim_task.checkpoint_data);
                    assert!(sim_task.action_applied);

                    let mut cnt = 0;
                    let mut _state;
                    let mut reward;
                    let mut done = false; // NOTE if already done, then this simulation will not be scheduled
                    let mut accu_reward = 0.0;
                    let mut accu_gamma = 1.0;
                    let mut _info;
                    // start_state_value = self.get_value(_state) // TODO
                    let start_state_value = 0.0; // to tune?
                    let factor = 1.0; //  to tune?
                    let mut rng = rand::thread_rng();

                    // Only calculate the action weights (and epsilon) once per simulation task
                    let mut action_weights_pruned: Vec<f32> = vec![0.0; action_n]; // 1.0 if the action will not lead to saturation
                    let mut action_weights_heavy: Vec<f32> = vec![0.0; action_n]; // (reward / visit count) if the action will not lead to saturation
                    let mut epsilon = 0.0;

                    // env loop
                    while !done {
                        if rollout_strategy.as_str() != "random" {
                            action_weights_pruned = sim_task.children_saturated.clone().iter().map(|&x| {
                                if x {
                                    0.0
                                } else {
                                    1.0
                                }
                            }).collect();
                        }
    
                        let mut non_saturated_actions_visited = 0.0;
                        if rollout_strategy.as_str() == "heavy" {
                            // Only consider rewards of actions that do not lead to saturation
                            for (visit_count, reward, saturated, action_weight) in izip!(&visit_counts, &rewards, &sim_task.children_saturated, &mut action_weights_heavy) {
                                if !saturated && *visit_count > 0.0 {
                                    *action_weight = reward / visit_count;
                                    non_saturated_actions_visited += 1.0;
                                }
                            }
    
                            // Max is necessary in case all actions lead to saturation
                            epsilon = f32::max(non_saturated_actions_visited / ((action_n - sim_task.children_saturated_cnt) as f32), 0.0);
                            // Epsilon = min((#non saturated actions with visit count > 0 / #non saturated actions); 0.75)
                            epsilon = f32::min(epsilon, 0.75);
                        }
                        
                        let action_weights = match rollout_strategy.as_str() {
                            // Random policy rollouts
                            "random" => {
                                vec![1.0; action_n]
                            },
                            // Random policy rollout with action pruning
                            "pruning" => {
                                // If all children are saturated, just pick a random action
                                if sim_task.children_saturated_cnt == action_n {
                                    vec![1.0; action_n]
                                } else {
                                    action_weights_pruned.clone()
                                }
                            },
                            // Heavy rollouts
                            "heavy" => {
                                // If all actions weights are zero, pick a random action
                                let rand = rng.gen_range(0.0..1.0);

                                if action_weights_heavy.iter().all(|&x| x == 0.0) || rand > epsilon {
                                    if sim_task.children_saturated_cnt == action_n {
                                        vec![1.0; action_n]
                                    } else {
                                        action_weights_pruned.clone()
                                    }
                                } else {
                                    action_weights_heavy.clone()
                                }
                            },
                            "lookahead" => {
                                // Make a copy of current env
                                let env_checkpoint = env.checkpoint().clone();
                                let lh_actions = rand::seq::index::sample(&mut rng, action_n, 5).into_vec();

                                let mut best_action = std::usize::MAX;
                                let mut best_reward = std::f32::MIN;

                                for lh_action in lh_actions {
                                    env.restore(env_checkpoint.clone());
                                    (_state, reward, _, _info) = env.step(lh_action);

                                    if reward > best_reward {
                                        best_action = lh_action;
                                        best_reward = reward;
                                    }
                                }

                                env.restore(env_checkpoint);

                                if best_action == std::usize::MAX {
                                    panic!("Something went wrong with lookahead search");
                                }
                                let mut action_weights = vec![0.0; action_n];
                                action_weights[best_action] = 1.0;
                                action_weights
                            },
                            _ => {
                                panic!("Unkown rollout strategy: {}", rollout_strategy);
                            }
                        };

                        let actions: Vec<usize> = (0..action_n).collect();
                        let dist = WeightedIndex::new(&action_weights).unwrap();
                        let action = actions[dist.sample(&mut rng)];

                        (_state, reward, done, _info) = env.step(action);
                        // This will only work properly with dense rewards!
                        visit_counts[action] += 1.0;
                        rewards[action] += reward;

                        // timeLimited truncate
                        if cnt == max_sim_step && !done {
                            done = true;
                            // get the final reward
                            // reward = env.get_reward();
                        }

                        accu_reward += reward * accu_gamma;
                        accu_gamma *= gamma;
                        cnt += 1;
                    }

                    //  Use V(s) to stabilize simulation return
                    accu_reward = accu_reward * factor + start_state_value * (1.0 - factor);

                    // reply
                    tx2.send(Reply::DoneSimulation(task_idx, accu_reward))
                        .unwrap();
                }

                Message::Nothing => {
                    // act as random straggler
                    let mut rng = rand::thread_rng();
                    thread::sleep(Duration::from_secs(rng.gen_range(0..5)));
                    tx2.send(Reply::OK).unwrap();
                }
            }
        }
    });

    (handle.unwrap(), tx, rx2)
}
