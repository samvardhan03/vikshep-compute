//! `vikshep-mcp`: the MCP data plane on stdin/stdout (`docs/mcp.md`).
//!
//! ```text
//! vikshep-mcp [--store shm|file] [--store-dir DIR] [--pad zero_pad|circular]
//!             [--executor local|cloud] [--keep-segments]
//! ```
//!
//! * `--store`: `shm` (POSIX shared memory, the default on Linux and macOS)
//!   or `file` (one file per OID in a directory, the default on Windows).
//! * `--store-dir`: directory of the file store (default
//!   `$VIKSHEP_SHM_DIR` or `<temp dir>/vikshep-shm`).
//! * `--pad`: pad policy of every axis (default `zero_pad`); the contract's
//!   `ScatterCfg` has no pad field.
//! * `--executor`: recorded in provenance manifests (default `local`).
//! * `--keep-segments`: do not remove the segments this server created when
//!   it exits (by default they are removed at end of input).
//!
//! Diagnostics go to stderr; stdout carries only JSON-RPC messages.

use std::io::{BufReader, stdin, stdout};
use std::path::PathBuf;
use std::process::ExitCode;

use vikshep_mcp::server::{Options, Server};
use vikshep_mcp::store::{FileStore, TensorStore, default_store};
use vikshep_numerics::provenance::Executor;
use vikshep_scatter::PadPolicy;

fn usage() -> ExitCode {
    eprintln!(
        "usage: vikshep-mcp [--store shm|file] [--store-dir DIR] [--pad zero_pad|circular] \
         [--executor local|cloud] [--keep-segments]"
    );
    ExitCode::from(2)
}

fn main() -> ExitCode {
    let mut opts = Options::default();
    let mut kind: Option<String> = None;
    let mut dir: Option<PathBuf> = None;
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let mut value = || it.next().cloned();
        match a.as_str() {
            "--store" => kind = value(),
            "--store-dir" => dir = value().map(PathBuf::from),
            "--pad" => match value().as_deref() {
                Some("zero_pad") => opts.pad = PadPolicy::ZeroPad,
                Some("circular") => opts.pad = PadPolicy::Circular,
                _ => return usage(),
            },
            "--executor" => match value().as_deref().and_then(Executor::parse) {
                Some(e) => opts.executor = e,
                None => return usage(),
            },
            "--keep-segments" => opts.keep_segments = true,
            _ => return usage(),
        }
    }
    let store: Result<Box<dyn TensorStore>, String> = match (kind.as_deref(), dir) {
        (Some("file"), d) => FileStore::new(&d.unwrap_or_else(FileStore::default_dir))
            .map(|s| Box::new(s) as Box<dyn TensorStore>),
        #[cfg(unix)]
        (Some("shm"), None) => Ok(Box::new(vikshep_mcp::store::ShmStore::new())),
        #[cfg(not(unix))]
        (Some("shm"), None) => {
            Err("POSIX shared memory is not available on this platform; use --store file".into())
        }
        (None, None) => default_store(),
        (None, Some(d)) => FileStore::new(&d).map(|s| Box::new(s) as Box<dyn TensorStore>),
        _ => return usage(),
    };
    let store = match store {
        Ok(s) => s,
        Err(e) => {
            eprintln!("vikshep-mcp: {e}");
            return ExitCode::FAILURE;
        }
    };
    eprintln!(
        "vikshep-mcp {}: store {}",
        env!("CARGO_PKG_VERSION"),
        store.kind()
    );
    let mut server = Server::new(store, opts);
    match server.serve(BufReader::new(stdin().lock()), stdout().lock()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            server.shutdown();
            eprintln!("vikshep-mcp: {e}");
            ExitCode::FAILURE
        }
    }
}
