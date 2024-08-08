#![allow(dead_code)]

use std::fs::OpenOptions;
use std::path::Path;
use std::io::Write;
use std::path::PathBuf;

use serde::Serialize;

pub fn save_data_to_file<T: Serialize>(data: &T, output_dir: &PathBuf, filename: &str) {
    let filename = Path::new(output_dir).join(filename);
    let mut file = OpenOptions::new().append(true).create(true).open(filename).unwrap();
    let serialized_data = serde_json::to_string(data).expect("Failed to convert json to string");
    if let Err(e) = writeln!(file, "{}", serialized_data) {
        eprintln!("Couldn't write to file: {}", e);
    }
}