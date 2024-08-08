extern crate bindgen;

use std::env;
use std::path::PathBuf;

fn main() {
    // Tell cargo to tell rustc to link the libraries.
    println!("cargo:rustc-link-search=/opt/conda/lib");
    println!("cargo:rustc-link-lib=protobuf");
    println!("cargo:rustc-link-search=/usr/local/lib");
    println!("cargo:rustc-link-lib=taso_runtime");

    // Tell cargo to invalidate the built crate whenever the wrapper changes
    println!("cargo:rerun-if-changed=wrapper.h");

    // The bindgen::Builder is the main entry point
    // to bindgen, and lets you build up options for
    // the resulting bindings.
    let bindings = bindgen::Builder::default()
        .enable_cxx_namespaces()
        .header("wrapper.h")
        .clang_args(&["-x", "c++", "-std=c++11"])
        .whitelist_type("std::map")
        .whitelist_type("std::set")
        .whitelist_type("std::vector")
        .whitelist_type("taso::Graph")
        .whitelist_type("taso::Tensor")
        .opaque_type("std::.*")
        // Tell cargo to invalidate the built crate whenever any of the
        // included header files changed.
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        // Finish the builder and generate the bindings.
        .generate()
        // Unwrap the Result and panic on failure.
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
