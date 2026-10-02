//! `vikshep-conformance`: run VDS-1 conformance cases.
//!
//! Usage:
//!
//! ```text
//! vikshep-conformance hashes          print the determinism-sweep hashes (JSON)
//! vikshep-conformance check           compare against conformance/vectors (exit 1 on mismatch)
//! ```

use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("hashes") => {
            print!("{}", vikshep_conformance::hashes_json());
            ExitCode::SUCCESS
        }
        Some("check") => {
            let got = vikshep_conformance::hashes_json();
            if got == vikshep_conformance::GOLDEN_HASHES_JSON {
                println!(
                    "conformance: all {} cases match",
                    vikshep_conformance::CASES.len()
                );
                ExitCode::SUCCESS
            } else {
                eprintln!("conformance: MISMATCH against conformance/vectors/v1/hashes.json");
                eprint!("{got}");
                ExitCode::FAILURE
            }
        }
        _ => {
            eprintln!("usage: vikshep-conformance <hashes|check>");
            ExitCode::from(2)
        }
    }
}
