//! Integration test: spawn `vikshep-mcp`, perform the MCP handshake, place a
//! conformance input tensor in the store by OID, and check that the output
//! OIDs equal the conformance vectors. Runs with the platform default store
//! (POSIX shared memory on Linux and macOS, the file store on Windows) and
//! with the file store everywhere.

use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use serde_json::{Value as Json, json};
use vikshep_conformance_core::suite::{embedded_expected_output, scatter_case};
use vikshep_mcp::store::{FileStore, TensorStore, default_store};
use vikshep_numerics::NUMERICS_VERSION;
use vikshep_numerics::oid::oid;

struct Client {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    next_id: i64,
}

impl Client {
    fn spawn(args: &[&str]) -> Self {
        let mut child = Command::new(env!("CARGO_BIN_EXE_vikshep-mcp"))
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .expect("spawn vikshep-mcp");
        let stdin = child.stdin.take().unwrap();
        let stdout = BufReader::new(child.stdout.take().unwrap());
        Self {
            child,
            stdin,
            stdout,
            next_id: 1,
        }
    }

    fn send(&mut self, msg: &Json) {
        writeln!(self.stdin, "{msg}").unwrap();
        self.stdin.flush().unwrap();
    }

    fn request(&mut self, method: &str, params: Json) -> Json {
        let id = self.next_id;
        self.next_id += 1;
        self.send(&json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params}));
        let mut line = String::new();
        self.stdout.read_line(&mut line).unwrap();
        let v: Json =
            serde_json::from_str(&line).unwrap_or_else(|e| panic!("bad response {line:?}: {e}"));
        assert_eq!(v["id"], id);
        assert_eq!(v["jsonrpc"], "2.0");
        v
    }

    /// Call a tool; return its structured summary (parsed from the text
    /// content) after checking the fields every response must carry.
    fn call(&mut self, name: &str, arguments: Json) -> Json {
        let v = self.request("tools/call", json!({"name": name, "arguments": arguments}));
        let r = &v["result"];
        let text = r["content"][0]["text"].as_str().unwrap().to_string();
        assert_eq!(r["isError"], false, "{name}: {text}");
        let s: Json = serde_json::from_str(&text).unwrap();
        assert_eq!(s["numerics_version"], NUMERICS_VERSION, "{name}");
        assert_eq!(s["manifest_hash"].as_str().unwrap().len(), 64, "{name}");
        assert_eq!(r["structuredContent"], s, "{name}");
        s
    }

    fn call_error(&mut self, name: &str, arguments: Json) -> String {
        let v = self.request("tools/call", json!({"name": name, "arguments": arguments}));
        assert_eq!(v["result"]["isError"], true, "{v}");
        v["result"]["content"][0]["text"]
            .as_str()
            .unwrap()
            .to_string()
    }

    fn handshake(&mut self) {
        let init = self.request(
            "initialize",
            json!({"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "test", "version": "0"}}),
        );
        assert_eq!(init["result"]["protocolVersion"], "2025-06-18");
        assert_eq!(init["result"]["serverInfo"]["name"], "vikshep-mcp");
        assert_eq!(
            init["result"]["_meta"]["numerics_version"],
            NUMERICS_VERSION
        );
        self.send(&json!({"jsonrpc": "2.0", "method": "notifications/initialized"}));
        let tools = self.request("tools/list", json!({}));
        let names: Vec<&str> = tools["result"]["tools"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t["name"].as_str().unwrap())
            .collect();
        assert_eq!(
            names,
            ["compute_scattering", "reduce", "compare", "detect_anomaly"]
        );
    }

    fn finish(mut self) {
        drop(self.stdin);
        let status = self.child.wait().unwrap();
        assert!(status.success());
    }
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn expected_oid(case: &str, output: &str) -> String {
    embedded_expected_output(case, output).unwrap().0[..28].to_string()
}

/// Manifest hash of the 1-D case below, pinned: equal on every platform.
const MANIFEST_HASH_1D: &str = "6ff71ab48d632625384a0f74aaa30f965fe3f06c444ea3e4ff9d4f2623a061cf";

fn exercise(client: &mut Client, store: &mut dyn TensorStore) {
    client.handshake();

    // 1-D conformance case (zero padding is the server default).
    let case = "scatter/1d/256/J4-Q1-L1/zero_pad/trivial/o2/uniform";
    let (_, x) = scatter_case(case).unwrap();
    let bytes = f32_bytes(&x);
    let input_oid = oid(&bytes);
    let created = store.write(&input_oid, &bytes).unwrap();
    let s = client.call(
        "compute_scattering",
        json!({"input_oid": input_oid, "signal_len": 256, "cfg": {"J": 4, "Q": 1}}),
    );
    let coeff_oid = s["coeff_oid"].as_str().unwrap().to_string();
    assert_eq!(
        coeff_oid,
        expected_oid(case, "S"),
        "coefficients OID equals the conformance vector"
    );
    assert_eq!(s["shape"], json!([1, 13, 16]));
    assert_eq!(s["store"], store.kind());
    assert_eq!(
        s["manifest_hash"], MANIFEST_HASH_1D,
        "manifest hash is platform independent"
    );
    // the output segment is readable by OID and hashes to it
    let n = 13 * 16 * 4;
    assert_eq!(oid(&store.read(&coeff_oid, n).unwrap()), coeff_oid);
    assert_eq!(s["manifest_oid"].as_str().unwrap().len(), 28);

    let r = client.call("reduce", json!({"coeff_oid": coeff_oid}));
    assert_eq!(r["method"], "ratio");
    assert_eq!(r["output_oid"], expected_oid(case, "r2"));
    let lm = client.call(
        "reduce",
        json!({"coeff_oid": coeff_oid, "method": "log_mean"}),
    );
    assert_eq!(lm["output_oid"], expected_oid(case, "log_mean"));
    assert_eq!(lm["shape"], json!([1, 13]));
    let alias = client.call(
        "reduce_scattering",
        json!({"coeff_oid": coeff_oid, "method": "mean"}),
    );
    let std = client.call("reduce", json!({"coeff_oid": coeff_oid, "method": "std"}));
    assert_ne!(alias["output_oid"], std["output_oid"]);

    // A small library of events with the same configuration, one outlier.
    let mut outlier = String::new();
    for i in 0..8u32 {
        let sig: Vec<f32> = (0..256u32)
            .map(|t| {
                let base = ((t * 37 + i * 11) % 101) as f32 / 101.0 - 0.5;
                if i == 7 { 40.0 * base } else { base }
            })
            .collect();
        let b = f32_bytes(&sig);
        let o = oid(&b);
        store.write(&o, &b).unwrap();
        let s = client.call(
            "compute_scattering",
            json!({"input_oid": o, "signal_len": 256, "cfg": {"J": 4, "Q": 1}}),
        );
        if i == 7 {
            outlier = s["coeff_oid"].as_str().unwrap().to_string();
        }
        store.remove(&o).unwrap();
    }
    let c = client.call("compare", json!({"query_oid": coeff_oid, "k": 3}));
    assert_eq!(c["library_size"], 8);
    let nb = c["neighbors"].as_array().unwrap();
    assert_eq!(nb.len(), 3);
    assert!(nb[0]["sw1"].as_f64().unwrap() <= nb[1]["sw1"].as_f64().unwrap());
    let d_normal = client.call(
        "detect_anomaly",
        json!({"query_oid": lm["output_oid"], "tau": 1.0, "k": 3}),
    );
    let d_out = client.call(
        "detect_anomaly",
        json!({"query_oid": outlier, "tau": 1.0, "k": 3}),
    );
    assert_eq!(d_out["is_anomaly"], true, "{d_out}");
    assert!(d_out["distance"].as_f64().unwrap() > 1.0);
    assert!(d_normal["distance"].as_f64().unwrap() < d_out["distance"].as_f64().unwrap());
    assert_eq!(d_normal["nearest_oid"].as_str().unwrap().len(), 28);

    // errors are tool errors, not crashes
    assert!(
        client
            .call_error("reduce", json!({"coeff_oid": "0".repeat(28)}))
            .contains("not a coefficient tensor")
    );
    assert!(
        client
            .call_error(
                "compute_scattering",
                json!({"input_oid": "1".repeat(28), "signal_len": 256, "cfg": {"J": 4, "Q": 1}})
            )
            .contains("1111")
    );
    assert!(client
        .call_error("compute_scattering", json!({"input_oid": input_oid, "signal_len": 256, "cfg": {"J": 2, "Q": 1, "dim": "3"}}))
        .contains("not supported"));
    assert!(
        client
            .call_error(
                "compute_scattering",
                json!({"input_oid": input_oid, "signal_len": 128, "cfg": {"J": 2, "Q": 1}})
            )
            .contains("hash")
    );

    if created {
        store.remove(&input_oid).unwrap();
    }
}

#[test]
fn default_store_session() {
    let mut store = default_store().unwrap();
    let mut client = Client::spawn(&[]);
    exercise(&mut client, store.as_mut());
    client.finish();
}

#[test]
fn file_store_session_and_cleanup() {
    let dir: PathBuf = std::env::temp_dir().join(format!("vikshep-mcp-it-{}", std::process::id()));
    let mut store = FileStore::new(&dir).unwrap();
    let mut client = Client::spawn(&["--store", "file", "--store-dir", dir.to_str().unwrap()]);
    exercise(&mut client, &mut store);
    client.finish();
    // the server removed every segment it created
    let left: Vec<_> = std::fs::read_dir(&dir).unwrap().collect();
    assert!(left.is_empty(), "{left:?}");
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn two_d_so2_session() {
    let dir: PathBuf = std::env::temp_dir().join(format!("vikshep-mcp-2d-{}", std::process::id()));
    let mut store = FileStore::new(&dir).unwrap();
    let mut client = Client::spawn(&["--store-dir", dir.to_str().unwrap(), "--pad", "circular"]);
    client.handshake();
    let case = "scatter/2d/32x32/J1-Q1-L4/circular.circular/so2_relative/o2/uniform";
    let (_, x) = scatter_case(case).unwrap();
    let b = f32_bytes(&x);
    let input_oid = oid(&b);
    store.write(&input_oid, &b).unwrap();
    let s = client.call(
        "compute_scattering",
        json!({"input_oid": input_oid, "signal_len": 1024,
               "cfg": {"J": 1, "Q": 1, "L": 4, "dim": "2", "group": "so2", "dim_shape": [32, 32]}}),
    );
    assert_eq!(s["coeff_oid"], expected_oid(case, "S"));
    assert_eq!(s["config"]["group"], "so2_relative");
    let r = client.call(
        "reduce",
        json!({"coeff_oid": s["coeff_oid"], "method": "log_mean"}),
    );
    assert_eq!(r["output_oid"], expected_oid(case, "log_mean"));
    store.remove(&input_oid).unwrap();
    client.finish();
    std::fs::remove_dir_all(&dir).unwrap();
}
