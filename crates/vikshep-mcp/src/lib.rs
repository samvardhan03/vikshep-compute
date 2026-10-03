//! The open MCP data plane of Vikshep (`docs/mcp.md`): a Model Context
//! Protocol server speaking JSON-RPC 2.0 over stdio, one message per line,
//! with the tools `compute_scattering`, `reduce`, `compare` and
//! `detect_anomaly`. Tensors cross process boundaries only as 28-hex OIDs
//! naming POSIX shared-memory segments (a file-backed store on Windows).
//!
//! This crate makes no network connections, collects no telemetry and has
//! no licence checks.

pub mod server;
pub mod store;
