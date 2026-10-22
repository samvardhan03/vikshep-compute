//! `vikshep-conformance`: run VDS-1 conformance cases.
//!
//! ```text
//! vikshep-conformance run [--backend cpu|cpu-serial|capi-cpu] [--vectors DIR]
//!                         [--report FILE] [--only PREFIX]...
//!     run conformance suite v1; print (or write) the JSON report;
//!     exit 1 if any case fails
//! vikshep-conformance generate [--backend B] [--vectors DIR]
//!     regenerate conformance/vectors/v3 with the CPU reference
//! vikshep-conformance list
//!     print every case id with its stream id
//! vikshep-conformance hashes
//!     print the C0 determinism-sweep hashes (JSON)
//! vikshep-conformance check
//!     compare the sweep hashes with conformance/vectors/v3/hashes.json
//! vikshep-conformance issue --manifest FILE [--backend NAME]
//!     issue a Determinism Certificate from a provenance manifest JSON file;
//!     runs the quick self-test and prints the certificate JSON to stdout
//! vikshep-conformance verify CERTIFICATE --inputs DIR
//!     re-run the computation recorded in CERTIFICATE with the open CPU
//!     reference and compare every output OID; print VERIFIED or MISMATCH
//! ```

use std::path::PathBuf;
use std::process::ExitCode;

use vikshep_backend_api::ScatterBackend;
use vikshep_conformance::suite;

fn usage() -> ExitCode {
    eprintln!(
        "usage: vikshep-conformance <run|generate|list|hashes|check|issue|verify> ...\n\
         \n\
         run      [--backend cpu|cpu-serial|capi-cpu] [--vectors DIR] [--report FILE] [--only PREFIX]\n\
         generate [--backend B] [--vectors DIR]\n\
         list\n\
         hashes\n\
         check\n\
         issue    --manifest FILE [--backend NAME]\n\
         verify   CERTIFICATE --inputs DIR"
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

    // issue and verify have their own argument shapes; handle them before
    // the generic `parse()` path.
    match cmd.as_str() {
        "issue" => return cmd_issue(&args[1..]),
        "verify" => return cmd_verify(&args[1..]),
        _ => {}
    }

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
                eprintln!("conformance: MISMATCH against conformance/vectors/v3/hashes.json");
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

/// `issue --manifest FILE [--backend NAME]`
fn cmd_issue(args: &[String]) -> ExitCode {
    let mut manifest_file: Option<PathBuf> = None;
    let mut backend_name = "cpu".to_string();
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--manifest" => {
                manifest_file = it.next().map(PathBuf::from);
            }
            "--backend" => {
                if let Some(b) = it.next() {
                    backend_name = b.clone();
                } else {
                    eprintln!("issue: --backend requires a value");
                    return ExitCode::from(2);
                }
            }
            other => {
                eprintln!("issue: unknown argument {other:?}");
                return ExitCode::from(2);
            }
        }
    }
    let Some(path) = manifest_file else {
        eprintln!("issue: --manifest FILE is required");
        return ExitCode::from(2);
    };
    let manifest_json = match std::fs::read_to_string(&path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("issue: cannot read {}: {e}", path.display());
            return ExitCode::FAILURE;
        }
    };
    match vikshep_conformance::certificate::issue(&manifest_json, &backend_name) {
        Ok(cert) => {
            println!("{cert}");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("issue failed: {e}");
            ExitCode::FAILURE
        }
    }
}

/// `verify CERTIFICATE --inputs DIR`
fn cmd_verify(args: &[String]) -> ExitCode {
    let mut cert_file: Option<PathBuf> = None;
    let mut inputs_dir: Option<PathBuf> = None;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--inputs" => {
                inputs_dir = it.next().map(PathBuf::from);
            }
            other if !other.starts_with('-') && cert_file.is_none() => {
                cert_file = Some(PathBuf::from(other));
            }
            other => {
                eprintln!("verify: unknown argument {other:?}");
                return ExitCode::from(2);
            }
        }
    }
    let (Some(cert_path), Some(inputs)) = (cert_file, inputs_dir) else {
        eprintln!("verify: usage: verify CERTIFICATE --inputs DIR");
        return ExitCode::from(2);
    };
    let cert_json = match std::fs::read_to_string(&cert_path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("verify: cannot read {}: {e}", cert_path.display());
            return ExitCode::FAILURE;
        }
    };
    let report = vikshep_conformance::certificate::verify(&cert_json, &inputs);
    if let Some(err) = &report.error {
        eprintln!("verify: {err}");
        return ExitCode::FAILURE;
    }
    for out in &report.outputs {
        if out.matches {
            println!("  output {}: {} OK", out.name, out.expected_oid);
        } else {
            println!(
                "  output {}: expected {} got {}  MISMATCH",
                out.name, out.expected_oid, out.computed_oid
            );
        }
    }
    if report.manifest_hash_ok {
        println!("  manifest_hash: OK");
    } else {
        println!("  manifest_hash: MISMATCH");
    }
    if report.verified {
        println!("VERIFIED");
        ExitCode::SUCCESS
    } else {
        println!("MISMATCH");
        ExitCode::FAILURE
    }
}
