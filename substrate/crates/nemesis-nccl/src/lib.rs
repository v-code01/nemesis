//! `nemesis-nccl` — NCCL communicator management for the NEMESIS control plane.
//!
//! Provides:
//! - [`backend::NcclBackend`]: async trait abstracting communicator operations.
//! - [`sim::NcclSim`]: deterministic simulated backend for tests and staging.
//! - [`real`]: real NCCL hardware backend (feature-gated on `cuda`).
//! - [`service::HealerServiceImpl`]: tonic gRPC service implementation.

pub mod backend;
pub mod real;
pub mod service;
pub mod sim;
