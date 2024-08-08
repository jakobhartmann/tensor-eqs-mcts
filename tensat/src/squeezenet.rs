use crate::{model::*, parse::*};
use egg::*;
use std::fs::*;

/// Gets the RecExpr of a inceptionv3 model
pub fn get_squeezenet() -> RecExpr<Mdl> {
    // Step 1: Read serialized model file
    let model_file = "/usr/tensat/model/squeezenet.model";
    let serialized =
        read_to_string(model_file).expect("Something went wrong reading the model file");

    // Step 2: parse to get model
    let graph = parse_model(&serialized);

    // Step 3: get the RexExpr
    graph.rec_expr()
}
