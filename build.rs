use std::env;

fn main() {
    // evil for OB.
    env::set_var("CC", "clang");
    env::set_var("CXX", "clang++");
    prep_net();
    build_dependencies();
    generate_bindings();
}

fn prep_net() {
    let net_path = env::var("EVALFILE").unwrap_or("viridithas.nnue.zst".into());
    if net_path == "viridithas.nnue.zst" {
        // check if net exists
        if let Err(e) = std::fs::metadata(net_path) {
            eprintln!("Couldn't read default net during build script! {e}");
            eprintln!("Note: viri looks for a default net in the project root called \"viridithas.nnue.zst\".");
        }
        return;
    }
    std::fs::copy(net_path, "viridithas.nnue.zst").unwrap();
}

fn build_dependencies() {
    #[cfg(feature = "syzygy")]
    build_fathom();
}

#[cfg(feature = "syzygy")]
fn build_fathom() {
    let cc = &mut cc::Build::new();
    cc.file("./deps/fathom/src/tbprobe.c");
    cc.include("./deps/fathom/src/");
    cc.define("_CRT_SECURE_NO_WARNINGS", None);

    // MSVC doesn't support stdatomic.h, so use clang on Windows
    if env::consts::OS == "windows" {
        cc.compiler("clang");
    }

    cc.compile("fathom");
}

fn generate_bindings() {
    #[cfg(all(feature = "bindgen", feature = "syzygy"))]
    generate_fathom_bindings();
}

#[cfg(all(feature = "bindgen", feature = "syzygy"))]
fn generate_fathom_bindings() {
    let bindings = bindgen::Builder::default()
        .header("./deps/fathom/src/tbprobe.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .layout_tests(false)
        .generate()
        .unwrap();

    bindings.write_to_file("./src/tablebases/bindings.rs").unwrap();
}
