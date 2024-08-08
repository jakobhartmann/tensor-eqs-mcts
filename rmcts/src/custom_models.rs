#![allow(unused_variables)]
#![allow(dead_code)]

use egg::*;
use tensat::model::*;
use tensat::input::GraphConverter;

pub fn get_very_simple_model() -> RecExpr<Mdl> {
    let mut graph = GraphConverter::default();

    let input = graph.new_input(&[1, 64, 56, 56]);

    // Weights
    let w1 = graph.new_weight(&[64, 64, 1, 1]);

    // Convolution
    let tmp = graph.conv2d(input, w1, 1, 1, PSAME, ACTNONE);

    // Explicit relu
    graph.relu(tmp);
    
    // Generate and return expression
    graph.rec_expr()
}

pub fn get_simple_model() -> RecExpr<Mdl> {
    let mut graph = GraphConverter::default();

    let input = graph.new_input(&[1, 64, 56, 56]);

    // Weights
    let w1 = graph.new_weight(&[64, 64, 1, 1]);
    let w2 = graph.new_weight(&[64, 64, 1, 1]);

    // Initial convolution
    let tmp = graph.conv2d(input, w1, 1, 1, PSAME, ACTNONE);
    let res1_input = graph.relu(tmp);

    // ResBlock
    let tmp = graph.conv2d(res1_input, w2, 1, 1, PSAME, ACTNONE);
    let res1_output = graph.add(res1_input, tmp);

    graph.relu(res1_output);
    
    // Generate and return expression
    graph.rec_expr()
}

pub fn get_complex_model() -> RecExpr<Mdl> {
    let mut graph = GraphConverter::default();

    let input = graph.new_input(&[1, 64, 56, 56]);

    // Weights
    let w1 = graph.new_weight(&[64, 64, 1, 1]);
    let w2 = graph.new_weight(&[64, 64, 1, 1]);
    let w3 = graph.new_weight(&[64, 64, 1, 1]);
    let w4 = graph.new_weight(&[64, 64, 1, 1]);
    let w5 = graph.new_weight(&[64, 64, 1, 1]);

    // Initial convolution
    let tmp = graph.conv2d(input, w1, 1, 1, PSAME, ACTNONE);
    let res1_input = graph.relu(tmp);

    // ResBlock 1
    let tmp = graph.conv2d(res1_input, w2, 1, 1, PSAME, ACTNONE);
    let tmp = graph.relu(tmp);
    let tmp = graph.conv2d(tmp, w3, 1, 1, PSAME, ACTNONE);
    let res1_output = graph.add(res1_input, tmp);

    let res2_input = graph.relu(res1_output);

    // ResBlock 2
    let tmp = graph.conv2d(res2_input, w4, 1, 1, PSAME, ACTNONE);
    let tmp = graph.relu(tmp);
    let tmp = graph.conv2d(tmp, w5, 1, 1, PSAME, ACTNONE);
    let res2_output = graph.add(res2_input, tmp);

    graph.relu(res2_output);
    
    // Generate and return expression
    graph.rec_expr()
}

pub fn get_complex_model2() -> RecExpr<Mdl> {
    let mut graph = GraphConverter::default();

    let input = graph.new_input(&[1, 64, 56, 56]);

    // Weights
    let w1 = graph.new_weight(&[64, 64, 1, 1]);
    let w2 = graph.new_weight(&[64, 64, 1, 1]);
    let w3 = graph.new_weight(&[64, 64, 1, 1]);
    let w4 = graph.new_weight(&[64, 64, 1, 1]);
    let w5 = graph.new_weight(&[64, 64, 1, 1]);
    let w6 = graph.new_weight(&[64, 64, 1, 1]);

    // Initial convolution
    let tmp = graph.conv2d(input, w1, 1, 1, PSAME, ACTNONE);
    let res1_input = graph.relu(tmp);

    // ResBlock 1
    let tmp = graph.conv2d(res1_input, w2, 1, 1, PSAME, ACTNONE);
    let tmp = graph.relu(tmp);
    let tmp = graph.conv2d(tmp, w3, 1, 1, PSAME, ACTNONE);
    let tmp2 = graph.conv2d(tmp, w4, 1, 1, PSAME, ACTNONE);
    let res1_output = graph.add(tmp, tmp2);

    let res2_input = graph.relu(res1_output);

    // ResBlock 2
    let tmp = graph.conv2d(res2_input, w5, 1, 1, PSAME, ACTNONE);
    let tmp = graph.relu(tmp);
    let tmp = graph.conv2d(tmp, w6, 1, 1, PSAME, ACTNONE);
    let res2_output = graph.add(res2_input, tmp);

    graph.relu(res2_output);
    
    // Generate and return expression
    graph.rec_expr()
}

pub fn get_suboptimal_model() -> RecExpr<Mdl> {
    let mut graph = GraphConverter::default();

    let input = graph.new_input(&[1, 64, 56, 56]);

    // Weights
    let w1 = graph.new_weight(&[64, 64, 1, 1]);
    let w2 = graph.new_weight(&[64, 64, 1, 1]);
    let w3 = graph.new_weight(&[64, 64, 1, 1]);

    // Initial convolution
    let tmp = graph.conv2d(input, w1, 1, 1, PSAME, ACTNONE);
    let tmp = graph.relu(tmp);

    let conv1 = graph.conv2d(tmp, w2, 1, 1, PSAME, ACTNONE);
    let conv2 = graph.conv2d(tmp, w3, 1, 1, PSAME, ACTNONE);

    let add1 = graph.add(conv1, conv2);
    let add2 = graph.add(conv2, conv1);

    let tmp = graph.concat(1, 4, add1, add2);

    // Generate and return expression
    graph.rec_expr()
}

pub fn get_138_model() -> RecExpr<Mdl> {
    let mut graph = GraphConverter::default();

    let input = graph.new_input(&[1, 64, 56, 56]);

    // Weights
    let w1 = graph.new_weight(&[64, 64, 1, 1]);
    let w2 = graph.new_weight(&[64, 64, 1, 1]);

    // Cconvolution
    let conv1 = graph.conv2d(input, w1, 1, 1, PSAME, ACTRELU);
    let conv2 = graph.conv2d(input, w2, 1, 1, PSAME, ACTRELU);

    let tmp = graph.noop(conv1, conv2);

    // Generate and return expression
    graph.rec_expr()
}