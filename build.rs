use std::env;

fn main() {
    prep_net();
    build_dependencies();
    generate_bindings();
    emit_git_info();
}

fn emit_git_info() {
    let git = |args: &[&str]| {
        std::process::Command::new("git")
            .args(args)
            .output()
            .ok()
            .filter(|out| out.status.success())
            .map(|out| String::from_utf8_lossy(&out.stdout).trim().to_owned())
    };

    let hash = git(&["rev-parse", "HEAD"]).unwrap_or_else(|| "unknown".to_owned());
    let dirty = git(&["status", "--porcelain"]).is_some_and(|s| !s.is_empty());

    println!("cargo:rustc-env=VIRIDITHAS_GIT_HASH={hash}");
    println!("cargo:rustc-env=VIRIDITHAS_GIT_DIRTY={}", u8::from(dirty));
}

fn prep_net() {
    let net_path = env::var("EVALFILE").unwrap_or_else(|_| "viridithas.nnue.zst".into());
    if net_path == "viridithas.nnue.zst" {
        // check if net exists
        if let Err(e) = std::fs::metadata(net_path) {
            eprintln!("Couldn't read default net during build script! {e}");
            eprintln!(
                "Note: viri looks for a zstd-compressed default net in the project root called \"viridithas.nnue.zst\"."
            );
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
    cc.file("./deps/pyrrhic/tbprobe.c");
    cc.include("./deps/pyrrhic/");
    cc.define("_CRT_SECURE_NO_WARNINGS", None);
    cc.cpp(false);
    cc.std("c17");

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
        .header("./deps/pyrrhic/tbprobe.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .layout_tests(false)
        .generate()
        .unwrap();

    bindings
        .write_to_file("./src/tablebases/bindings.rs")
        .unwrap();
}
