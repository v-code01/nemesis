//! Analytic bandwidth model for common GPU interconnects.
//!
//! All bandwidth values are in GB/s (gigabytes per second, 10^9 bytes/s).
//! Latency modelling is omitted — for NCCL all-reduce over large tensors
//! the bandwidth term dominates (alpha-beta model with large beta*n).
//!
//! This module is intentional public API for scenario-level bandwidth
//! analysis and is not yet wired into the main simulation loop.

#![allow(dead_code)]

pub struct BandwidthModel {
    pub nvlink_a100_gbps: f64,
    pub pcie_gen4_gbps:   f64,
    pub ib_hdr_1hop_gbps: f64,
    pub ib_hdr_2hop_gbps: f64,
}

impl Default for BandwidthModel {
    fn default() -> Self {
        Self {
            nvlink_a100_gbps:  600.0,
            pcie_gen4_gbps:     64.0,
            ib_hdr_1hop_gbps:  200.0,
            ib_hdr_2hop_gbps:  160.0,
        }
    }
}

impl BandwidthModel {
    /// Ring all-reduce time in nanoseconds.
    ///
    /// Formula: 2 * (N-1)/N * bytes / bandwidth_bytes_per_ns
    /// Derived from the standard ring all-reduce algorithm where each GPU
    /// sends and receives 2*(N-1)/N * bytes over N steps.
    pub fn all_reduce_ns(&self, bytes: u64, n_gpus: u32, bw_gbps: f64) -> u64 {
        // GB/s to bytes/ns: 1 GB/s = 1 byte/ns (10^9 / 10^9), divide by 8 for bits→bytes
        let bw_bytes_per_ns = bw_gbps / 8.0;
        let alpha = 2.0 * (n_gpus - 1) as f64 / n_gpus as f64;
        (alpha * bytes as f64 / bw_bytes_per_ns) as u64
    }

    /// Effective Model FLOP Utilization (MFU) accounting for communication overhead.
    ///
    /// MFU = compute_time / (compute_time + comms_time)
    /// A value of 1.0 means no communication overhead; practical values for
    /// NVLink-full A100 clusters run ~0.45-0.55 for large LLM training steps.
    pub fn effective_mfu(
        &self,
        compute_flops:  f64,
        peak_tflops:    f64,
        gradient_bytes: u64,
        n_gpus:         u32,
        bw_gbps:        f64,
    ) -> f64 {
        let compute_ns = (compute_flops / (peak_tflops * 1e12)) * 1e9;
        let comms_ns   = self.all_reduce_ns(gradient_bytes, n_gpus, bw_gbps) as f64;
        compute_ns / (compute_ns + comms_ns)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nvlink_mfu_higher_than_pcie() {
        let bw = BandwidthModel::default();
        let mfu_nvl  = bw.effective_mfu(14e12, 100.0, 14_000_000_000, 8, bw.nvlink_a100_gbps);
        let mfu_pcie = bw.effective_mfu(14e12, 100.0, 14_000_000_000, 8, bw.pcie_gen4_gbps);
        assert!(
            mfu_nvl > mfu_pcie * 1.4,
            "NVLink MFU {mfu_nvl:.3} should be >1.4x PCIe MFU {mfu_pcie:.3}"
        );
    }
}
