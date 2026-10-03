//! Records the target triple for provenance manifests
//! (`vikshep_numerics::provenance::PLATFORM`).

fn main() {
    let target = std::env::var("TARGET").unwrap_or_else(|_| "unknown".into());
    println!("cargo:rustc-env=VIKSHEP_TARGET={target}");
    println!("cargo:rerun-if-changed=build.rs");
}
