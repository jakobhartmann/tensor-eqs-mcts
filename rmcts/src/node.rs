use rand::{thread_rng};
use rand::distributions::{WeightedIndex, Distribution};
use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt;
use std::rc::Rc;


#[allow(dead_code)]
pub fn print_tree(
    node: &Node, 
    max_depth: usize, 
    depth: usize,
)
{
    if depth >= max_depth {
        return
    }

    println!("{}Action: {:?}, checkpoint idx: {}, is head: {}, #children saturated: {}, #children pruned: {}, visit count: {}, visited node count: {}, updated node count: {}", " ".repeat(3 * depth), node.action, node.checkpoint_idx, node.is_head, node.children_saturated_cnt, node.children_pruned, node.visit_count, node.visited_node_count, node.updated_node_count);
    
    for child_node in &node.children {
        match child_node {
            Some(child_node) => {
                print_tree(&child_node.borrow(), max_depth, depth + 1);
            },
            None => {},
        }
    }
}

#[derive(Debug, Clone)]
pub struct Node {
    pub action: Option<usize>,
    pub action_n: usize,
    pub checkpoint_idx: u32,
    pub parent: Option<Rc<RefCell<Node>>>,
    pub gamma: f32,
    pub is_head: bool,

    // children
    pub children: Vec<Option<Rc<RefCell<Node>>>>,
    pub rewards: Vec<f32>,
    pub dones: Vec<bool>,
    children_visit_count: Vec<u32>,
    children_complete_visit_count: Vec<u32>,
    pub children_saturated: Vec<bool>,
    pub children_saturated_cnt: usize,
    pub children_pruned: usize,
    q_value: Vec<f32>,

    // self
    pub visit_count: u32,
    traverse_history: HashMap<u32, (usize, f32)>,
    visited_node_count: usize,
    updated_node_count: usize,
}

#[allow(unused_must_use)]
impl fmt::Display for Node {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Is head: {}, children_saturated_cnt: {}, children_pruned: {}, visit_count: {}, visited node count: {}, updated node count: {}", self.is_head, self.children_saturated_cnt, self.children_pruned, self.visit_count, self.visited_node_count, self.updated_node_count);
        write!(f, ", rewards: {:?}, dones: {:?}, children visit count: {:?}, children complete visit count: {:?}, children saturated: {:?}, q value: {:?}", self.rewards, self.dones, self.children_visit_count, self.children_complete_visit_count, self.children_saturated, self.q_value)
    }
}

impl Node {
    pub fn new(
        action: Option<usize>,
        action_n: usize,
        checkpoint_idx: u32,
        gamma: f32,
        is_head: bool,
        parent: Option<Rc<RefCell<Node>>>,
        children_saturated: Vec<bool>,
        children_saturated_cnt: usize,
    ) -> Rc<RefCell<Self>> {
        Rc::new(RefCell::new(Node {
            action: action,
            action_n: action_n,
            checkpoint_idx: checkpoint_idx,
            parent: parent,
            gamma: gamma,
            is_head: is_head,
            children: vec![None; action_n],
            rewards: vec![0.0; action_n],
            dones: vec![false; action_n],
            children_visit_count: vec![0; action_n],
            children_complete_visit_count: vec![0; action_n],
            children_saturated: children_saturated,
            children_saturated_cnt: children_saturated_cnt,
            children_pruned: children_saturated_cnt,
            q_value: vec![0.0; action_n],
            visit_count: 0,
            traverse_history: HashMap::new(),
            visited_node_count: 0,
            updated_node_count: 0,
        }))
    }

    pub fn dummy() -> Rc<RefCell<Self>> {
        Rc::new(RefCell::new(Node {
            action: None,
            action_n: 0,
            checkpoint_idx: 0,
            parent: None,
            gamma: 1.0,
            is_head: false,
            children: vec![None; 1],
            rewards: vec![0.0; 1],
            dones: vec![false; 1],
            children_visit_count: vec![0; 1],
            children_complete_visit_count: vec![0; 1],
            children_saturated: vec![false; 1],
            children_saturated_cnt: 0,
            children_pruned: 0,
            q_value: vec![0.0; 1],
            visit_count: 0,
            traverse_history: HashMap::new(),
            visited_node_count: 0,
            updated_node_count: 0,
        }))
    }

    pub fn all_child_visited(&self) -> bool {
        self.visited_node_count == self.action_n
    }

    pub fn no_child_available(&self) -> bool {
        (self.updated_node_count == 0) || (self.updated_node_count == (self.children_saturated_cnt - self.children_pruned))
    }

    pub fn all_child_saturated(&self) -> bool {
        self.children_saturated_cnt == self.action_n
    }

    pub fn all_non_pruned_child_visited(&self) -> bool {
        (self.visited_node_count + self.children_pruned) == self.action_n
    }

    // pub fn all_child_updated(&self) -> bool {
    //     self.updated_node_count == self.action_n
    // }

    pub fn shallow_clone(&self) -> NodeStub {
        NodeStub {
            action_n: self.action_n,
            is_head: self.is_head,
            // NOTE: only children_visit_count is up-to-date with all selected action!
            children_visit_count: self.children_visit_count.clone(),
            // NOTE: needed for select_expansion_action to select a non-saturated node! Only works with sequential version!
            children_saturated: self.children_saturated.clone(),
        }
    }

    pub fn select_uct_action(&self, max: bool) -> usize {
        let mut best_score = std::f32::MIN;
        let mut best_action = std::usize::MAX;
        let mut sat_count = 0;
        let mut child_missing_count = 0;
        for action in 0..(self.action_n as usize) {
            if self.children[action].is_none() {
                child_missing_count += 1;
                continue;
            }
            if self.children_saturated[action] {
                sat_count += 1;
                continue;
            }

            let exploit_score =
                self.q_value[action] / (self.children_complete_visit_count[action] as f32);
            // let explore_score = f32::sqrt(
            //     2.0 * f32::ln(self.visit_count as f32) / (self.children_visit_count[action] as f32),
            // );
            // TODO consider std?
            let explore_score = if max {
                0.0
            } else {
                f32::sqrt(
                    2.0 * f32::ln(self.visit_count as f32)
                        / (self.children_visit_count[action] as f32),
                )
            };
            let score = exploit_score + 2.0 * explore_score;

            if score > best_score {
                best_score = score;
                best_action = action;
            }
        }
        if best_action == std::usize::MAX {
            panic!(
                "Can't select uct action, all children are saturated or none! {} - {} - {} - {}",
                self.is_head, self.updated_node_count, sat_count, child_missing_count
            );
        }
        if max {
            println!(
                "best_action {} and expected score {}",
                best_action, best_score
            );
        }
        best_action
    }

    pub fn select_max_visited_action(&self) -> usize {
        let mut max_visits = std::u32::MIN;
        let mut best_action = std::usize::MAX;
        let mut sat_count = 0;
        let mut child_missing_count = 0;
        for action in 0..(self.action_n as usize) {
            if self.children[action].is_none() {
                child_missing_count += 1;
                continue;
            }
            if self.children_saturated[action] {
                sat_count += 1;
                continue;
            }

            let visits = self.children_visit_count[action];

            if visits > max_visits {
                max_visits = visits;
                best_action = action;
            }
        }
        if best_action == std::usize::MAX {
            panic!(
                "Can't select uct action, all children are saturated or none! {} - {} - {} - {}",
                self.is_head, self.updated_node_count, sat_count, child_missing_count
            );
        }
        best_action
    }

    pub fn update_history(&mut self, idx: u32, action_taken: usize, reward: f32) {
        self.traverse_history.insert(idx, (action_taken, reward));
    }

    pub fn add_child(
        &mut self,
        expand_action: usize,
        saving_idx: u32,
        gamma: f32,
        child_saturated: bool,
        children_saturated: Vec<bool>,
        children_saturated_cnt: usize,
        self_node: Rc<RefCell<Node>>,
    ) {
        if child_saturated {
            self.children_saturated[expand_action] = true;
            self.children_saturated_cnt += 1;
        }

        match &self.children[expand_action] {
            None => {
                self.children[expand_action] = Some(Node::new(
                    Some(expand_action),
                    self.action_n,
                    saving_idx,
                    gamma,
                    false,
                    Some(self_node),
                    children_saturated,
                    children_saturated_cnt,
                ))
            }
            Some(child) => panic!(
                "self {} - action {} - to-add {} - existing child {}",
                self.checkpoint_idx,
                expand_action,
                saving_idx,
                child.borrow().checkpoint_idx
            ),
        }
    }

    pub fn update_incomplete(&mut self, idx: u32) {
        let (action_taken, _) = self.traverse_history.get(&idx).unwrap().clone();
        if self.children_visit_count[action_taken] == 0 {
            self.visited_node_count += 1;
        }
        self.children_visit_count[action_taken] += 1;
        self.visit_count += 1;
    }

    pub fn update_complete(&mut self, idx: u32, accu_reward: f32) -> f32 {
        let (action_taken, reward);
        match self.traverse_history.get(&idx) {
            Some((a, r)) => {
                action_taken = a.clone();
                reward = r.clone();
            }
            None => panic!("no item {} - {}", self.checkpoint_idx, idx),
        }
        let this_accu_reward = reward + self.gamma * accu_reward;
        if self.children_complete_visit_count[action_taken] == 0 {
            self.updated_node_count += 1
        }
        self.children_complete_visit_count[action_taken] += 1;
        self.q_value[action_taken] += this_accu_reward;
        this_accu_reward
    }
}

// impl PartialEq for Node {
//     fn eq(&self, other: &Self) -> bool {
//         self.checkpoint_idx == other.checkpoint_idx
//     }
// }
//
// impl Eq for Node {}

#[derive(Debug, Clone)]
pub struct NodeStub {
    pub action_n: usize,
    pub is_head: bool,
    pub children_visit_count: Vec<u32>,
    pub children_saturated: Vec<bool>,
}

impl NodeStub {
    pub fn select_expansion_action(&self) -> usize {
        let mut action_weights = vec![0.0; self.action_n];
        for action in 0..self.action_n {
            if self.children_visit_count[action] == 0 && !self.children_saturated[action] {
                action_weights[action] = 1.0;
            }
        }

        if action_weights.iter().all(|&x| x == 0.0) {
            panic!("Cannot select expansion action, because all children are either saturated or have already been visited!");
        }

        let actions: Vec<usize> = (0..self.action_n).collect();
        let dist = WeightedIndex::new(&action_weights).unwrap();
        let mut rng = thread_rng();
        actions[dist.sample(&mut rng)]


        // let mut cnt = 0;
        // let mut rng = rand::thread_rng();
        // let mut action: usize = 0;

        // loop {
        //     if cnt < 20 {
        //         action = rng.gen_range(0..self.action_n);
        //     }

        //     if cnt > 100 {
        //         return action;
        //     }

        //     if self.children_visit_count[action] == 0 {
        //         return action;
        //     }

        //     if self.children_visit_count[action] > 0 && cnt < 10 {
        //         cnt += 1;
        //         continue;
        //     }
        // }
    }
}
