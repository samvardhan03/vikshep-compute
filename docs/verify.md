# How a reviewer verifies a Vikshep result

A Vikshep Determinism Certificate lets you confirm — without trusting any
Vikshep server or binary — that a claimed computation was actually performed
and that the outputs match.

## What is a certificate?

A certificate is a JSON file that contains three things:

1. **A provenance manifest** — the complete description of the computation:
   what operation was run, what the inputs were (identified by their
   cryptographic OIDs), what configuration was used, what the output OIDs are,
   and which software version and platform produced them.  The manifest also
   contains a hash of its own content (`manifest_hash`) that detects any
   modification.

2. **Conformance evidence** — a record that the issuing installation passed the
   VDS-1 conformance self-test at the time of issuance.  This confirms that
   the software is conforming to the determinism specification, not a
   customised build.

3. **An optional signature** — an Ed25519 signature that proves *which
   installation* issued the certificate.  You do not need to verify the
   signature to verify the computation; signatures identify the issuer, they do
   not prove correctness.

## What does "verify" mean?

Verification means re-running the exact same computation from scratch using
the open CPU reference implementation and confirming that the output bytes
match.  Because the algorithm is bit-exact across all conforming backends and
platforms (see the VDS-1 specification), if your CPU re-run produces the same
output OIDs as the certificate records, the outputs are genuine.

This is a proof of computation, not just a proof of integrity.  You are not
trusting a signature or a hash; you are independently reproducing the result.

## Step-by-step verification

### Requirements

- A checkout of this repository (`vikshep-compute`).
- Rust toolchain (version pinned in `rust-toolchain.toml`).
- The certificate file (e.g. `result.cert.json`).
- The raw input tensors in a directory.  Each tensor is stored as
  `<oid>.bin`, where `<oid>` is the 28-character lowercase hex OID that
  appears in the certificate manifest.  The issuing party must provide these
  files alongside the certificate.

### Build the verifier

```sh
cargo build --release -p vikshep-conformance
```

The binary is at `target/release/vikshep-conformance`.

### Run verification

```sh
vikshep-conformance verify result.cert.json --inputs ./inputs/
```

The verifier:

1. Reads and parses the certificate.
2. Validates the `manifest_hash` to ensure the manifest body has not been
   altered.
3. Loads each input tensor from `./inputs/<oid>.bin`.
4. Re-runs the operation (e.g. wavelet scattering) with the CPU reference.
5. Computes the OID of the output and compares it to the OID in the
   certificate.

Output on success:

```
  output coefficients: <oid>  OK
  manifest_hash: OK
VERIFIED
```

Output on any mismatch:

```
  output coefficients: expected <oid_a>  got <oid_b>  MISMATCH
  manifest_hash: OK
MISMATCH
```

Exit code is `0` for VERIFIED, non-zero for MISMATCH or any error.

### Verify from Python

If you have the `vikshep-compute` Python package installed:

```python
import vikshep_compute, pathlib, json

cert_json = pathlib.Path("result.cert.json").read_text()
result = vikshep_compute.verify_certificate(cert_json, "./inputs/")

if result["verified"]:
    print("VERIFIED")
else:
    print("MISMATCH")
    for o in result["outputs"]:
        status = "OK" if o["matches"] else "MISMATCH"
        print(f"  {o['name']}: {status}")
```

## What the certificate does NOT prove

- The certificate does not prove that the *inputs* are authentic.  The inputs
  are identified by OID (a cryptographic hash of their bytes), so any
  alteration to the input data will change the OID, causing verification to
  fail at the file-load step.  Provenance of the inputs is the responsibility
  of the issuing party.
- An unsigned certificate does not prove which software installation produced
  it.  The optional Ed25519 signature addresses this for installations that
  enrol a device key.
- The Geant4 simulation that produced the input data is Tier-3 (outside the
  scope of VDS-1); its reproducibility depends on the Geant4 version, build,
  platform and seeds recorded in the provenance manifest.

## Certificate schema

The JSON schema is at `spec/certificate.schema.json`.  The embedded manifest
follows `spec/provenance.schema.json`.

## Conformance CI

Every push to this repository runs the full VDS-1 conformance suite and
publishes cross-platform hash comparison results.  The CI status reflects
whether the reference implementation itself is conforming.
