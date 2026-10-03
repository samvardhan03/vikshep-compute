//! The MCP server: JSON-RPC 2.0 messages, one per line, and the tools
//! `compute_scattering`, `reduce`, `compare` and `detect_anomaly`
//! (`docs/mcp.md`). Tool inputs mirror the public Vikshep contract
//! (`contract/mcpSchemas.ts`): tensors are referenced by OID, results are
//! written to the tensor store under their OIDs, and responses carry only
//! OIDs and small JSON summaries, always with `numerics_version` and the
//! provenance `manifest_hash`.

use std::collections::BTreeMap;
use std::io::{BufRead, Write};
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{Map, Value as Json, json};
use vikshep_anomaly::graph::{AnomalyConfig, AnomalyIndex};
use vikshep_anomaly::sw1::{Cloud, fingerprint_clouds};
use vikshep_cpu::CpuBackend;
use vikshep_numerics::jcs::{Value, object};
use vikshep_numerics::oid::oid;
use vikshep_numerics::provenance::{Dtype, Execution, Executor, Manifest, TensorRef};
use vikshep_numerics::{NUMERICS_VERSION, TIER2_VERSION};
use vikshep_scatter::manifest::{config_value, scattering_manifest};
use vikshep_scatter::reduce::{log_mean, path_mean, path_std, r2};
use vikshep_scatter::{Group, PadPolicy, ScatterConfig, ScatterOutput, Scattering};

use crate::store::{TensorStore, is_oid};

/// MCP protocol revisions this server speaks, newest first.
pub const PROTOCOL_VERSIONS: &[&str] = &["2025-06-18", "2025-03-26", "2024-11-05"];

/// Server options.
#[derive(Clone, Debug)]
pub struct Options {
    /// Pad policy of every axis (the contract has no pad field).
    pub pad: PadPolicy,
    /// Executor recorded in manifests.
    pub executor: Executor,
    /// Leave the segments this server created in place when it exits.
    pub keep_segments: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            pad: PadPolicy::ZeroPad,
            executor: Executor::Local,
            keep_segments: false,
        }
    }
}

/// What the server knows about a tensor it produced.
enum Entry {
    /// Coefficients of one `compute_scattering` call.
    Coefficients {
        sc: Box<Scattering>,
        out: ScatterOutput,
    },
    /// A reduction of a coefficient tensor.
    Reduction { source: String },
    /// Any other output (manifests, distance lists).
    Other,
}

/// A tool failure, reported as an MCP tool result with `isError: true`.
struct ToolError(String);

impl<E: std::fmt::Display> From<E> for ToolError {
    fn from(e: E) -> Self {
        Self(e.to_string())
    }
}

type ToolResult = Result<Value, ToolError>;

fn err<T>(msg: impl Into<String>) -> Result<T, ToolError> {
    Err(ToolError(msg.into()))
}

/// The server state.
pub struct Server {
    store: Box<dyn TensorStore>,
    opts: Options,
    protocol: String,
    /// Tensors produced in this session, by OID.
    registry: BTreeMap<String, Entry>,
    /// Coefficient OIDs in order of first computation (the reference
    /// library of `compare` and `detect_anomaly`).
    library: Vec<String>,
    /// Segments this server created (removed at shutdown).
    created: Vec<String>,
}

fn unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
}

fn shape_value(shape: &[usize]) -> Value {
    Value::Array(shape.iter().map(|&n| Value::Int(n as i64)).collect())
}

fn f64_bytes(v: &[f64]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// JSON-RPC error object.
fn rpc_error(id: &Json, code: i64, message: &str) -> Json {
    json!({"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}})
}

fn meta() -> Json {
    json!({"numerics_version": NUMERICS_VERSION, "tier2_version": TIER2_VERSION})
}

/// Integer argument with a default and bounds.
fn int_arg(
    args: &Map<String, Json>,
    key: &str,
    default: Option<i64>,
    lo: i64,
    hi: i64,
) -> Result<i64, ToolError> {
    match args.get(key) {
        None | Some(Json::Null) => {
            default.ok_or_else(|| ToolError(format!("missing required argument {key:?}")))
        }
        Some(v) => {
            let n = v
                .as_i64()
                .or_else(|| {
                    v.as_f64()
                        .filter(|f| f.fract() == 0.0 && f.abs() < 9.0e15)
                        .map(|f| f as i64)
                })
                .ok_or_else(|| ToolError(format!("{key} must be an integer")))?;
            if n < lo || n > hi {
                return err(format!("{key} must be in {lo}..={hi}"));
            }
            Ok(n)
        }
    }
}

fn str_arg<'a>(
    args: &'a Map<String, Json>,
    key: &str,
    default: Option<&'a str>,
    allowed: &[&str],
) -> Result<&'a str, ToolError> {
    let s = match args.get(key) {
        None | Some(Json::Null) => {
            default.ok_or_else(|| ToolError(format!("missing required argument {key:?}")))?
        }
        Some(v) => v
            .as_str()
            .ok_or_else(|| ToolError(format!("{key} must be a string")))?,
    };
    if !allowed.is_empty() && !allowed.contains(&s) {
        return err(format!("{key} must be one of {allowed:?}"));
    }
    Ok(s)
}

fn oid_arg<'a>(args: &'a Map<String, Json>, key: &str) -> Result<&'a str, ToolError> {
    let s = str_arg(args, key, None, &[])?;
    if !is_oid(s) {
        return err(format!("{key} must be exactly 28 lowercase hex characters"));
    }
    Ok(s)
}

/// JSON Schemas of the tool inputs (mirroring `contract/mcpSchemas.ts`).
fn tool_list() -> Json {
    let oid = json!({"type": "string", "pattern": "^[0-9a-f]{28}$", "description": "28-hex OID of a tensor in shared memory"});
    json!([
        {
            "name": "compute_scattering",
            "description": "Wavelet scattering coefficients (VDS-1 section 14) of the float32 tensor input_oid (signal_len samples) in shared memory. Writes the float32 coefficient tensor [batch, paths, out...] under its OID and returns coeff_oid, its shape, numerics_version and the provenance manifest_hash. Supported in numerics_version 1: dim 1 and 2, order 1 and 2, groups trivial and so2; dim 3, order 3 and so3 are rejected.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "input_oid": oid,
                    "signal_len": {"type": "integer", "exclusiveMinimum": 0},
                    "cfg": {
                        "type": "object",
                        "properties": {
                            "J": {"type": "integer", "minimum": 1, "maximum": 14},
                            "Q": {"type": "integer", "minimum": 1, "maximum": 32},
                            "L": {"type": "integer", "minimum": 1, "maximum": 16, "default": 1},
                            "order": {"type": "integer", "minimum": 1, "maximum": 3, "default": 2},
                            "dim": {"type": "string", "enum": ["1", "2", "3"], "default": "1"},
                            "group": {"type": "string", "enum": ["trivial", "so2", "so3"], "default": "trivial"},
                            "dim_shape": {"type": "array", "items": {"type": "integer", "exclusiveMinimum": 0}, "default": []}
                        },
                        "required": ["J", "Q"]
                    }
                },
                "required": ["input_oid", "signal_len", "cfg"]
            }
        },
        {
            "name": "reduce",
            "description": "Reduce a coefficient tensor produced by compute_scattering in this session: ratio (r2, float32 [batch, pairs, out...]), log_mean, mean or std (float64 [batch, paths]). Writes the result under its OID; returns output_oid, numerics_version and manifest_hash.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "coeff_oid": oid,
                    "method": {"type": "string", "enum": ["mean", "std", "log_mean", "ratio"], "default": "ratio"}
                },
                "required": ["coeff_oid"]
            }
        },
        {
            "name": "compare",
            "description": "The k nearest events (Sliced Wasserstein-1 between fingerprint distributions, deterministic HNSW; VDS-1 section 19) to the single-event coefficient tensor query_oid (or a reduction of it) among the coefficient tensors computed earlier in this session with the same configuration.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query_oid": oid,
                    "k": {"type": "integer", "minimum": 1, "maximum": 1000, "default": 10}
                },
                "required": ["query_oid"]
            }
        },
        {
            "name": "detect_anomaly",
            "description": "Flags query_oid when its Sliced Wasserstein-1 distance to the k-th nearest reference event exceeds tau (VDS-1 section 19.4); references are the coefficient tensors computed earlier in this session with the same configuration. Returns is_anomaly, distance (to the k-th neighbour), nearest_oid.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query_oid": oid,
                    "tau": {"type": "number", "exclusiveMinimum": 0},
                    "k": {"type": "integer", "minimum": 1, "maximum": 1000, "default": 10}
                },
                "required": ["query_oid", "tau"]
            }
        }
    ])
}

impl Server {
    /// A server over `store`.
    #[must_use]
    pub fn new(store: Box<dyn TensorStore>, opts: Options) -> Self {
        Self {
            store,
            opts,
            protocol: PROTOCOL_VERSIONS[0].into(),
            registry: BTreeMap::new(),
            library: Vec::new(),
            created: Vec::new(),
        }
    }

    /// Serve line-delimited messages from `input` until end of input, then
    /// remove the segments this server created (unless kept).
    pub fn serve(&mut self, input: impl BufRead, mut output: impl Write) -> std::io::Result<()> {
        for line in input.lines() {
            let line = line?;
            if let Some(resp) = self.handle_line(&line) {
                writeln!(output, "{resp}")?;
                output.flush()?;
            }
        }
        self.shutdown();
        Ok(())
    }

    /// Remove created segments (unless `keep_segments`).
    pub fn shutdown(&mut self) {
        if !self.opts.keep_segments {
            for o in std::mem::take(&mut self.created) {
                let _ = self.store.remove(&o);
            }
        }
    }

    /// Handle one line; `None` for notifications and blank lines.
    pub fn handle_line(&mut self, line: &str) -> Option<String> {
        if line.trim().is_empty() {
            return None;
        }
        let msg: Json = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(_) => return Some(rpc_error(&Json::Null, -32700, "parse error").to_string()),
        };
        match msg {
            Json::Array(items) if !items.is_empty() => {
                let out: Vec<Json> = items.iter().filter_map(|m| self.handle(m)).collect();
                (!out.is_empty()).then(|| Json::Array(out).to_string())
            }
            m => self.handle(&m).map(|r| r.to_string()),
        }
    }

    fn handle(&mut self, msg: &Json) -> Option<Json> {
        let Some(obj) = msg.as_object() else {
            return Some(rpc_error(&Json::Null, -32600, "invalid request"));
        };
        let id = obj.get("id").cloned();
        let method = obj.get("method").and_then(Json::as_str);
        let (Some(method), true) = (method, obj.get("jsonrpc") == Some(&json!("2.0"))) else {
            // a response from the client, or malformed
            return id.map(|i| rpc_error(&i, -32600, "invalid request"));
        };
        let params = obj.get("params").cloned().unwrap_or(Json::Null);
        let Some(id) = id else {
            return None; // notification: initialized, cancelled, ...
        };
        let result = match method {
            "initialize" => Ok(self.initialize(&params)),
            "ping" => Ok(json!({})),
            "tools/list" => Ok(json!({"tools": tool_list(), "_meta": meta()})),
            "tools/call" => self.tools_call(&params),
            _ => Err((-32601, format!("method not found: {method}"))),
        };
        Some(match result {
            Ok(r) => json!({"jsonrpc": "2.0", "id": id, "result": r}),
            Err((code, m)) => rpc_error(&id, code, &m),
        })
    }

    fn initialize(&mut self, params: &Json) -> Json {
        let requested = params.get("protocolVersion").and_then(Json::as_str);
        self.protocol = requested
            .filter(|v| PROTOCOL_VERSIONS.contains(v))
            .unwrap_or(PROTOCOL_VERSIONS[0])
            .to_string();
        json!({
            "protocolVersion": self.protocol,
            "capabilities": {"tools": {"listChanged": false}},
            "serverInfo": {"name": "vikshep-mcp", "version": env!("CARGO_PKG_VERSION")},
            "instructions": "Deterministic Vikshep compute (VDS-1). Tensors are passed by 28-hex OID in shared memory; results are written under their OIDs and returned as OIDs with small summaries.",
            "_meta": meta()
        })
    }

    fn tools_call(&mut self, params: &Json) -> Result<Json, (i64, String)> {
        let name = params
            .get("name")
            .and_then(Json::as_str)
            .ok_or((-32602, "tools/call needs a tool name".to_string()))?;
        let empty = Map::new();
        let args = match params.get("arguments") {
            None | Some(Json::Null) => &empty,
            Some(Json::Object(m)) => m,
            Some(_) => return Err((-32602, "arguments must be an object".into())),
        };
        let res = match name {
            "compute_scattering" => self.compute_scattering(args),
            // the public agent's recipes call the reduce step reduce_scattering
            "reduce" | "reduce_scattering" => self.reduce(args),
            "compare" => self.compare(args),
            "detect_anomaly" => self.detect_anomaly(args),
            other => return Err((-32602, format!("unknown tool: {other}"))),
        };
        Ok(match res {
            Ok(summary) => {
                let text = summary
                    .canonical()
                    .unwrap_or_else(|e| format!("{{\"error\":\"{e}\"}}"));
                let mut r = json!({
                    "content": [{"type": "text", "text": text}],
                    "isError": false,
                    "_meta": meta()
                });
                if self.protocol.as_str() >= "2025-06-18" {
                    r["structuredContent"] = serde_json::from_str(&text).unwrap_or(Json::Null);
                }
                r
            }
            Err(ToolError(m)) => json!({
                "content": [{"type": "text", "text": m}],
                "isError": true,
                "_meta": meta()
            }),
        })
    }

    /// Store an output tensor and remember it.
    fn put(&mut self, bytes: &[u8], entry: Entry) -> Result<String, ToolError> {
        let o = oid(bytes);
        if self.store.write(&o, bytes)? {
            self.created.push(o.clone());
        }
        self.registry.entry(o.clone()).or_insert(entry);
        Ok(o)
    }

    fn execution(&self, started: u64) -> Execution {
        let mut ex = Execution::local("cpu", env!("CARGO_PKG_VERSION"));
        ex.executor = self.opts.executor;
        ex.started_unix_ms = Some(started);
        ex.finished_unix_ms = Some(unix_ms());
        ex
    }

    /// Store the manifest JSON and add `manifest_hash`, `manifest_oid`,
    /// `numerics_version` and `tier2_version` to a summary.
    fn finish(&mut self, m: &Manifest, mut summary: Vec<(&'static str, Value)>) -> ToolResult {
        let json = m.to_json()?;
        let hash = m.manifest_hash()?;
        let moid = self.put(json.as_bytes(), Entry::Other)?;
        summary.push(("manifest_hash", Value::Str(hash)));
        summary.push(("manifest_oid", Value::Str(moid)));
        summary.push(("numerics_version", Value::Int(i64::from(NUMERICS_VERSION))));
        summary.push(("tier2_version", Value::Int(i64::from(TIER2_VERSION))));
        Ok(object(summary))
    }

    fn compute_scattering(&mut self, args: &Map<String, Json>) -> ToolResult {
        let started = unix_ms();
        let input_oid = oid_arg(args, "input_oid")?.to_string();
        let signal_len = int_arg(args, "signal_len", None, 1, i64::from(u32::MAX))? as usize;
        let cfg = match args.get("cfg") {
            Some(Json::Object(m)) => m,
            None => return err("missing required argument \"cfg\""),
            Some(_) => return err("cfg must be an object"),
        };
        let j = int_arg(cfg, "J", None, 1, 14)? as u32;
        let q = int_arg(cfg, "Q", None, 1, 32)? as u32;
        let l = int_arg(cfg, "L", Some(1), 1, 16)? as u32;
        let order = int_arg(cfg, "order", Some(2), 1, 3)? as u32;
        let dim = str_arg(cfg, "dim", Some("1"), &["1", "2", "3"])?;
        let group = str_arg(cfg, "group", Some("trivial"), &["trivial", "so2", "so3"])?;
        let dim_shape: Vec<usize> = match cfg.get("dim_shape") {
            None | Some(Json::Null) => Vec::new(),
            Some(Json::Array(a)) => a
                .iter()
                .map(|v| v.as_u64().filter(|&n| n > 0).map(|n| n as usize))
                .collect::<Option<_>>()
                .ok_or_else(|| ToolError("dim_shape must hold positive integers".into()))?,
            Some(_) => return err("dim_shape must be an array"),
        };
        if dim == "3" || group == "so3" || order == 3 {
            return err(
                "dim 3, order 3 and group so3 are not supported by numerics_version 1 (see STATUS.md)",
            );
        }
        let (shape, group) = match dim {
            "1" => {
                if group != "trivial" {
                    return err("group so2 requires dim 2");
                }
                let n = match dim_shape.as_slice() {
                    [] => signal_len,
                    [n] => *n,
                    _ => return err("dim 1 takes dim_shape [] or [length]"),
                };
                (vec![n], Group::Trivial)
            }
            _ => {
                let [r, c] = dim_shape.as_slice() else {
                    return err("dim 2 needs dim_shape [rows, cols]");
                };
                let g = if group == "so2" {
                    Group::So2Relative
                } else {
                    Group::Trivial
                };
                (vec![*r, *c], g)
            }
        };
        let per: usize = shape.iter().product();
        if !signal_len.is_multiple_of(per) {
            return err(format!(
                "signal_len {signal_len} is not a multiple of the signal size {per}"
            ));
        }
        let config = ScatterConfig {
            dim: shape.len(),
            group,
            j,
            q,
            l,
            max_order: order,
            pad: vec![self.opts.pad; shape.len()],
            shape,
            carrier_cutoff: vikshep_scatter::config::DEFAULT_CARRIER_CUTOFF,
        };
        let sc = Scattering::new(config).map_err(|e| ToolError(e.0))?;
        let bytes = self.store.read(&input_oid, signal_len * 4)?;
        let x: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let out = sc.run(&CpuBackend::new(), &x)?;
        let coeff_bytes = out.canonical_bytes();
        let batch = out.batch;
        let n_paths = out.paths.len();
        let mut shape = vec![batch, n_paths];
        shape.extend(&out.out_shape);
        let manifest_cfg = config_value(sc.config());
        let fbank = sc.filters().fingerprint();
        let coeff_oid = oid(&coeff_bytes);
        let m = scattering_manifest(&sc, batch, &input_oid, &coeff_oid, self.execution(started));
        let new = !self.registry.contains_key(&coeff_oid);
        self.put(
            &coeff_bytes,
            Entry::Coefficients {
                sc: Box::new(sc),
                out,
            },
        )?;
        if new {
            self.library.push(coeff_oid.clone());
        }
        self.finish(
            &m,
            vec![
                ("batch", Value::Int(batch as i64)),
                ("coeff_oid", Value::Str(coeff_oid)),
                ("config", manifest_cfg),
                ("dtype", Value::Str("float32".into())),
                ("filter_bank_sha3", Value::Str(fbank)),
                ("n_paths", Value::Int(n_paths as i64)),
                ("shape", shape_value(&shape)),
                ("store", Value::Str(self.store.kind().into())),
            ],
        )
    }

    fn coefficients_of(&self, o: &str) -> Result<(&Scattering, &ScatterOutput, String), ToolError> {
        let mut key = o.to_string();
        if let Some(Entry::Reduction { source }) = self.registry.get(o) {
            key = source.clone();
        }
        match self.registry.get(&key) {
            Some(Entry::Coefficients { sc, out }) => Ok((sc, out, key)),
            _ => err(format!(
                "{o} is not a coefficient tensor (or a reduction of one) produced by compute_scattering in this server session"
            )),
        }
    }

    fn reduce(&mut self, args: &Map<String, Json>) -> ToolResult {
        let started = unix_ms();
        let coeff_oid = oid_arg(args, "coeff_oid")?.to_string();
        let method = str_arg(
            args,
            "method",
            Some("ratio"),
            &["mean", "std", "log_mean", "ratio"],
        )?
        .to_string();
        let (sc, out, key) = self.coefficients_of(&coeff_oid)?;
        if key != coeff_oid {
            return err("coeff_oid must be a coefficient tensor, not a reduction");
        }
        let (bytes, dtype, shape, cfg) = match method.as_str() {
            "ratio" => {
                let cutoff = sc.config().carrier_cutoff;
                let rr = r2(out, cutoff);
                let mut shape = vec![rr.batch, rr.pairs.len()];
                shape.extend(&out.out_shape);
                let cfg = object([
                    ("carrier_cutoff", Value::Int(i64::from(cutoff))),
                    ("method", Value::Str(method.clone())),
                ]);
                (rr.canonical_bytes(), Dtype::Float32, shape, cfg)
            }
            m => {
                let v = match m {
                    "log_mean" => log_mean(out),
                    "mean" => path_mean(out),
                    _ => path_std(out),
                };
                let cfg = object([("method", Value::Str(method.clone()))]);
                (
                    f64_bytes(&v),
                    Dtype::Float64,
                    vec![out.batch, out.paths.len()],
                    cfg,
                )
            }
        };
        let in_shape = {
            let mut s = vec![out.batch, out.paths.len()];
            s.extend(&out.out_shape);
            s
        };
        let out_oid = oid(&bytes);
        let m = Manifest {
            operation: "reduce".into(),
            inputs: vec![TensorRef::new(
                "coefficients",
                &coeff_oid,
                Dtype::Float32,
                &in_shape,
            )],
            config: cfg,
            filter_bank_sha3: None,
            seeds: Vec::new(),
            outputs: vec![TensorRef::new(&method, &out_oid, dtype, &shape)],
            execution: self.execution(started),
        };
        self.put(
            &bytes,
            Entry::Reduction {
                source: coeff_oid.clone(),
            },
        )?;
        self.finish(
            &m,
            vec![
                ("coeff_oid", Value::Str(coeff_oid)),
                ("dtype", Value::Str(dtype.name().into())),
                ("method", Value::Str(method)),
                ("output_oid", Value::Str(out_oid)),
                ("shape", shape_value(&shape)),
            ],
        )
    }

    /// Query cloud, reference clouds with their `(oid, event)` labels, and
    /// the manifest inputs, for `compare` and `detect_anomaly`.
    #[allow(clippy::type_complexity)]
    fn neighbourhood(
        &self,
        query_oid: &str,
    ) -> Result<(Cloud, Vec<Cloud>, Vec<(String, usize)>, Vec<TensorRef>), ToolError> {
        let (sc, q, qkey) = self.coefficients_of(query_oid)?;
        if q.batch != 1 {
            return err("the query must hold exactly one event (batch 1)");
        }
        let shape_of = |o: &ScatterOutput| {
            let mut s = vec![o.batch, o.paths.len()];
            s.extend(&o.out_shape);
            s
        };
        let qcloud = fingerprint_clouds(q)?.remove(0);
        let mut inputs = vec![TensorRef::new("query", &qkey, Dtype::Float32, &shape_of(q))];
        let mut refs = Vec::new();
        let mut labels = Vec::new();
        for o in &self.library {
            if *o == qkey {
                continue;
            }
            if let Some(Entry::Coefficients { sc: rsc, out }) = self.registry.get(o) {
                if rsc.config() != sc.config() {
                    continue;
                }
                for (e, c) in fingerprint_clouds(out)?.into_iter().enumerate() {
                    refs.push(c);
                    labels.push((o.clone(), e));
                }
                inputs.push(TensorRef::new(
                    "reference",
                    o,
                    Dtype::Float32,
                    &shape_of(out),
                ));
            }
        }
        Ok((qcloud, refs, labels, inputs))
    }

    fn search(&mut self, args: &Map<String, Json>, detect: bool) -> ToolResult {
        let started = unix_ms();
        let query_oid = oid_arg(args, "query_oid")?.to_string();
        let k = int_arg(args, "k", Some(10), 1, 1000)? as usize;
        let tau = if detect {
            match args.get("tau").and_then(Json::as_f64) {
                Some(t) if t > 0.0 && t.is_finite() => t,
                Some(_) => return err("tau must be a positive number"),
                None => return err("missing required argument \"tau\""),
            }
        } else {
            0.0
        };
        let (qcloud, refs, labels, inputs) = self.neighbourhood(&query_oid)?;
        let config = AnomalyConfig {
            k,
            tau,
            ..AnomalyConfig::default()
        };
        let nn = if refs.is_empty() {
            Vec::new()
        } else {
            AnomalyIndex::build(&refs, config)?.knn(&qcloud, k)?
        };
        let dists: Vec<f64> = nn.iter().map(|x| x.1).collect();
        let bytes = f64_bytes(&dists);
        let dist_oid = oid(&bytes);
        let mut cfg = config.value();
        if !detect && let Value::Object(m) = &mut cfg {
            m.remove("tau");
        }
        let m = Manifest {
            operation: if detect { "detect_anomaly" } else { "compare" }.into(),
            inputs,
            config: cfg,
            filter_bank_sha3: None,
            seeds: vec![("hnsw".into(), config.hnsw.seed)],
            outputs: vec![TensorRef::new(
                "distances",
                &dist_oid,
                Dtype::Float64,
                &[dists.len()],
            )],
            execution: self.execution(started),
        };
        self.put(&bytes, Entry::Other)?;
        let label = |i: usize| &labels[nn[i].0];
        let mut summary = vec![
            ("distances_oid", Value::Str(dist_oid)),
            ("k", Value::Int(k as i64)),
            ("library_size", Value::Int(refs.len() as i64)),
            ("query_oid", Value::Str(query_oid)),
        ];
        if detect {
            let kth = (nn.len() >= k).then(|| nn[k - 1].1);
            summary.push(("distance", kth.map_or(Value::Null, Value::Num)));
            summary.push(("is_anomaly", Value::Bool(kth.is_some_and(|d| d > tau))));
            summary.push((
                "nearest_distance",
                nn.first().map_or(Value::Null, |x| Value::Num(x.1)),
            ));
            summary.push((
                "nearest_event",
                nn.first()
                    .map_or(Value::Null, |_| Value::Int(label(0).1 as i64)),
            ));
            summary.push((
                "nearest_oid",
                nn.first()
                    .map_or(Value::Null, |_| Value::Str(label(0).0.clone())),
            ));
            summary.push(("tau", Value::Num(tau)));
        } else {
            let neighbors = (0..nn.len())
                .map(|i| {
                    object([
                        ("event", Value::Int(label(i).1 as i64)),
                        ("oid", Value::Str(label(i).0.clone())),
                        ("sw1", Value::Num(nn[i].1)),
                    ])
                })
                .collect();
            summary.push(("neighbors", Value::Array(neighbors)));
        }
        self.finish(&m, summary)
    }

    fn compare(&mut self, args: &Map<String, Json>) -> ToolResult {
        self.search(args, false)
    }

    fn detect_anomaly(&mut self, args: &Map<String, Json>) -> ToolResult {
        self.search(args, true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::FileStore;

    fn server() -> (Server, std::path::PathBuf) {
        let dir = std::env::temp_dir().join(format!(
            "vikshep-mcp-unit-{}-{}",
            std::process::id(),
            unix_ms()
        ));
        let store = FileStore::new(&dir).unwrap();
        (Server::new(Box::new(store), Options::default()), dir)
    }

    #[test]
    fn json_rpc_framing() {
        let (mut s, dir) = server();
        assert!(s.handle_line("not json").unwrap().contains("-32700"));
        assert!(
            s.handle_line(r#"{"jsonrpc":"2.0","id":1,"method":"nope"}"#)
                .unwrap()
                .contains("-32601")
        );
        assert_eq!(
            s.handle_line(r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#),
            None
        );
        assert_eq!(s.handle_line("   "), None);
        let init = s
            .handle_line(r#"{"jsonrpc":"2.0","id":"a","method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"t","version":"0"}}}"#)
            .unwrap();
        let v: Json = serde_json::from_str(&init).unwrap();
        assert_eq!(v["result"]["protocolVersion"], "2025-03-26");
        assert_eq!(v["id"], "a");
        let batch = s
            .handle_line(
                r#"[{"jsonrpc":"2.0","id":2,"method":"ping"},{"jsonrpc":"2.0","method":"x"}]"#,
            )
            .unwrap();
        assert_eq!(
            serde_json::from_str::<Json>(&batch)
                .unwrap()
                .as_array()
                .unwrap()
                .len(),
            1
        );
        let list: Json = serde_json::from_str(
            &s.handle_line(r#"{"jsonrpc":"2.0","id":3,"method":"tools/list"}"#)
                .unwrap(),
        )
        .unwrap();
        let names: Vec<&str> = list["result"]["tools"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t["name"].as_str().unwrap())
            .collect();
        assert_eq!(
            names,
            ["compute_scattering", "reduce", "compare", "detect_anomaly"]
        );
        let unknown = s
            .handle_line(r#"{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"x"}}"#)
            .unwrap();
        assert!(unknown.contains("-32602"));
        let bad = s
            .handle_line(r#"{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"compute_scattering","arguments":{"input_oid":"zz","signal_len":4,"cfg":{"J":1,"Q":1}}}}"#)
            .unwrap();
        let v: Json = serde_json::from_str(&bad).unwrap();
        assert_eq!(v["result"]["isError"], true);
        std::fs::remove_dir_all(dir).unwrap();
    }
}
