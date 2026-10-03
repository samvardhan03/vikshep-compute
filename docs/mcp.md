# MCP data plane (`vikshep-mcp`)

`vikshep-mcp` is the open data plane that Vikshep's agent talks to. It speaks
JSON-RPC 2.0 over stdio using the Model Context Protocol, one message per
line (stdout carries only protocol messages; diagnostics go to stderr). Raw
tensors never travel in messages: they are passed by OID in shared memory,
and responses carry OIDs and small JSON summaries only. The binary makes no
network connections and collects no telemetry.

```sh
cargo build --release -p vikshep-mcp
target/release/vikshep-mcp [--store shm|file] [--store-dir DIR] \
    [--pad zero_pad|circular] [--executor local|cloud] [--keep-segments]
```

## Protocol

* `initialize`: the server answers with the client's `protocolVersion` if it
  supports it (`2025-06-18`, `2025-03-26`, `2024-11-05`), otherwise with
  `2025-06-18`; capabilities `{"tools": {"listChanged": false}}`.
* `notifications/initialized` and other notifications: no response.
* `ping`, `tools/list`, `tools/call`. Batches (JSON arrays) are accepted.
* Errors: `-32700` parse error, `-32600` invalid request, `-32601` unknown
  method, `-32602` unknown tool. Failures inside a tool (bad arguments, a
  missing tensor, an unsupported configuration) are tool results with
  `isError: true` and a message.
* Every result carries `_meta: {numerics_version, tier2_version}`. A
  successful tool result has one text content item holding the summary as
  RFC 8785 canonical JSON, and (protocol `2025-06-18`) the same object as
  `structuredContent`. Every summary includes `numerics_version`,
  `tier2_version`, `manifest_hash` (SHA3-256 of the provenance manifest's
  hashed subset, `spec/provenance.schema.json`) and `manifest_oid` (the
  manifest JSON itself, stored like any tensor).

## Tools

Input field names mirror `contract/mcpSchemas.ts` of the public Vikshep
repository (commit 7882dfc).

| Tool | Input | Output summary |
|---|---|---|
| `compute_scattering` | `input_oid`, `signal_len`, `cfg {J, Q, L = 1, order = 2, dim = "1", group = "trivial", dim_shape = []}` | `coeff_oid`, `dtype` (`float32`), `shape` `[batch, paths, out...]`, `batch`, `n_paths`, `config`, `filter_bank_sha3`, `store` |
| `reduce` | `coeff_oid`, `method` (`ratio` default, `log_mean`, `mean`, `std`) | `output_oid`, `method`, `dtype`, `shape`, `coeff_oid` |
| `compare` | `query_oid`, `k = 10` | `neighbors [{oid, event, sw1}]`, `library_size`, `distances_oid` |
| `detect_anomaly` | `query_oid`, `tau`, `k = 10` | `is_anomaly`, `distance`, `nearest_oid`, `nearest_event`, `nearest_distance`, `library_size`, `distances_oid` |

`compute_scattering` reads `signal_len` float32 values (little-endian) from
`input_oid`. In 1-D the signal shape is `[signal_len]`, or `dim_shape = [n]`
with `signal_len` a multiple of `n` (a batch); in 2-D `dim_shape = [rows,
cols]` and `signal_len` is a multiple of `rows * cols`. `group = "so2"` is the
`so2_relative` pooling of VDS-1 section 14.5. The contract has no pad field:
every axis uses the server's `--pad` policy (default `zero_pad`), recorded in
the manifest configuration. Not supported in `numerics_version` 1: `dim =
"3"`, `order = 3`, `group = "so3"`, and values the specification excludes
(for example `J > 11`, odd `L` in 2-D, `L != 1` or `Q > 1` in the wrong
dimension); such requests return a tool error.

`reduce`: `ratio` is r2 (VDS-1 section 14.8, float32 `[batch, pairs,
out...]`), `log_mean` the log-mean fingerprint (14.9), `mean` and `std` the
per-path mean and population standard deviation over output positions
(14.9), float64 `[batch, paths]`. The public agent's recipes call this step
`reduce_scattering`; the server accepts that name as an alias of `reduce`.

`compare` and `detect_anomaly` take a single-event coefficient tensor (or a
reduction of one, which stands for its coefficients). Their reference
library is every coefficient tensor computed earlier in the same server
session with the same configuration, each event separately, in order of
first computation, excluding the query itself. Distances are Sliced
Wasserstein-1 between fingerprint distributions with 32 directions, searched
in a deterministic HNSW index (VDS-1 section 19, seed 0). `detect_anomaly`
flags the query when its distance to the `k`-th nearest reference exceeds
`tau`; `distance` is that `k`-th distance (`null`, and not flagged, with
fewer than `k` references), `nearest_oid` the coefficient tensor of the
nearest reference.

`reduce`, `compare` and `detect_anomaly` resolve their inputs in the
session's registry: they accept OIDs produced by this server process, which
knows their configuration and shape. A coefficient tensor from another
process must first be recomputed with `compute_scattering`.

## Tensor store

A tensor lives in a segment named by its OID, the lowercase hex of the first
14 bytes of SHA3-256 of its raw bytes (VDS-1 section 9.2). The segment's
first bytes are the tensor, row-major and little-endian; dtype and shape
travel in the messages. A segment may be longer than the tensor (macOS
rounds shared memory up to whole pages); the reader takes the length implied
by the message and verifies that those bytes hash to the OID, so a wrong
length or corrupted data is reported, never used.

**POSIX shared memory** (default on Linux and macOS): object `/<oid>` (29
characters, within macOS's 31-character limit), created with `shm_open` and
mode `0600`. A Python producer writes a float32 input with

```python
import hashlib
from multiprocessing import shared_memory
buf = signal.astype("<f4").tobytes()
oid = hashlib.sha3_256(buf).digest()[:14].hex()
sm = shared_memory.SharedMemory(name=oid, create=True, size=len(buf))
sm.buf[: len(buf)] = buf
```

and reads an output with `shared_memory.SharedMemory(name=coeff_oid)`
(before Python 3.13, attaching registers the segment with Python's resource
tracker, which warns at exit; pass `track=False` on 3.13 and later).

**File-backed store** (default on Windows, which has no POSIX shared memory;
`--store file` anywhere): one file per tensor, named `<oid>`, holding the raw
bytes, in `$VIKSHEP_SHM_DIR` or `<temp dir>/vikshep-shm` (`--store-dir`
overrides). The server writes a temporary file and renames it, so readers
never see a partial tensor.

Lifetime: the server removes the segments it created when its input ends
(the agent closes stdin), unless `--keep-segments` is given. Inputs written
by other processes are never removed by the server.

## Provenance

Each tool call builds a manifest (`spec/provenance.schema.json`): the
operation, input and output tensors by OID with dtype and shape, the
configuration, the filter-bank SHA3 for scattering, both versions, seeds,
and the execution record (backend `cpu`, platform triple, executor from
`--executor`, wall-clock start and end). The execution record is not part of
`manifest_hash`, so the same request gives the same hash on every platform.
The integration test (`crates/vikshep-mcp/tests/stdio.rs`) pins the hash of
one request and checks the output OIDs against conformance suite v1, with
POSIX shared memory on Linux and macOS and the file store on every platform.
