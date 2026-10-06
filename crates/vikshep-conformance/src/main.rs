//! `vikshep-conformance`: run VDS-1 conformance cases.
//!
//! ```text
//! vikshep-conformance run [--backend cpu|cpu-serial|capi-cpu] [--vectors DIR]
//!                         [--report FILE] [--only PREFIX]...
//!     run conformance suite v1; print (or write) the JSON report;
//!     exit 1 if any case fails
//! vikshep-conformance generate [--backend B] [--vectors DIR]
//!     regenerate conformance/vectors/v2 with the CPU reference
//! vikshep-conformance list
//!     print every case id with its stream id
//! vikshep-conformance hashes
//!     print the C0 determinism-sweep hashes (JSON)
//! vikshep-conformance check
//!     compare the sweep hashes with conformance/vectors/v2/hashes.json
//! ```

use std::path::PathBuf;
use std::process::ExitCode;

use vikshep_backend_api::ScatterBackend;
use vikshep_conformance::suite;

fn usage() -> ExitCode {
    eprintln!(
        "usage: vikshep-conformance <run|generate|list|hashes|check> \
         [--backend cpu|cpu-serial|capi-cpu] [--vectors DIR] [--report FILE] [--only PREFIX]"
    );
    ExitCode::from(2)
}

struct Opts {
    backend: String,
    vectors: PathBuf,
    report: Option<PathBuf>,
    only: Vec<String>,
}

fn parse(args: &[String]) -> Option<Opts> {
    let mut o = Opts {
        backend: "cpu".into(),
        vectors: suite::default_vectors_dir(),
        report: None,
        only: Vec::new(),
    };
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--backend" => o.backend = it.next()?.clone(),
            "--vectors" => o.vectors = PathBuf::from(it.next()?),
            "--report" => o.report = Some(PathBuf::from(it.next()?)),
            "--only" => o.only.push(it.next()?.clone()),
            _ => return None,
        }
    }
    Some(o)
}

fn backend(name: &str) -> Option<Box<dyn ScatterBackend>> {
    match name {
        "cpu" => Some(Box::new(vikshep_cpu::CpuBackend::new())),
        "cpu-serial" => Some(Box::new(vikshep_cpu::CpuBackend::serial())),
        "capi-cpu" => Some(Box::new(vikshep_capi::cpu_through_c_abi())),
        _ => None,
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(cmd) = args.first() else {
        return usage();
    };
    let Some(opts) = parse(&args[1..]) else {
        return usage();
    };
    match cmd.as_str() {
        "hashes" => {
            print!("{}", vikshep_conformance::hashes_json());
            ExitCode::SUCCESS
        }
        "check" => {
            let got = vikshep_conformance::hashes_json();
            if got == vikshep_conformance::GOLDEN_HASHES_JSON {
                println!(
                    "conformance: all {} sweep hashes match",
                    vikshep_conformance::CASES.len()
                );
                ExitCode::SUCCESS
            } else {
                eprintln!("conformance: MISMATCH against conformance/vectors/v2/hashes.json");
                eprint!("{got}");
                ExitCode::FAILURE
            }
        }
        "list" => {
            for c in suite::Suite::load().cases {
                println!("{:016x} {}", c.stream_id, c.id);
            }
            ExitCode::SUCCESS
        }
        "generate" => {
            let Some(b) = backend(&opts.backend) else {
                return usage();
            };
            match suite::generate(b.as_ref(), &opts.vectors) {
                Ok(e) => {
                    eprintln!(
                        "generated {} cases into {}",
                        e.cases.len(),
                        opts.vectors.display()
                    );
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    eprintln!("generate failed: {e}");
                    ExitCode::FAILURE
                }
            }
        }
        "run" => {
            let Some(b) = backend(&opts.backend) else {
                return usage();
            };
            let report = match suite::run(b.as_ref(), &opts.vectors, &opts.only) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("run failed: {e}");
                    return ExitCode::FAILURE;
                }
            };
            let json = report.to_json();
            match &opts.report {
                Some(p) => std::fs::write(p, &json).expect("write report"),
                None => print!("{json}"),
            }
            eprintln!(
                "conformance ({}): {} cases, {} pass, {} fail",
                report.backend, report.summary.cases, report.summary.pass, report.summary.fail
            );
            if report.summary.fail == 0 && report.summary.cases > 0 {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            }
        }
        _ => usage(),
    }
}
