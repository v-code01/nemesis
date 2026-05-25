use crate::config::{ClusterSpec, SimConfig, TopologyKind};
use nemesis_graph::{ClusterGraph, LinkKind};
use nemesis_proto::telemetry::v1::MetricSample;
use parking_lot::RwLock;
use std::sync::Arc;

pub struct SimCluster {
    pub graph:   Arc<RwLock<ClusterGraph>>,
    pub gpu_ids: Vec<String>,
    config:      ClusterSpec,
    seed:        u64,
}

impl SimCluster {
    pub fn from_config(cfg: &SimConfig) -> anyhow::Result<Self> {
        let spec = &cfg.cluster;
        anyhow::ensure!(spec.gpu_count > 0, "gpu_count must be > 0, got {}", spec.gpu_count);
        let mut g = ClusterGraph::new();
        let gpu_ids: Vec<String> = (0..spec.gpu_count).map(|i| format!("gpu-{i}")).collect();
        for id in &gpu_ids {
            g.add_gpu(id, "node-0", 0);
        }
        match spec.topology {
            TopologyKind::NvlinkFull => {
                for i in 0..spec.gpu_count {
                    for j in (i + 1)..spec.gpu_count {
                        g.add_link(&gpu_ids[i], &gpu_ids[j], LinkKind::NvLink, spec.nvlink_gbps, 0);
                    }
                }
            }
            TopologyKind::PcieCrossNode => {
                for i in 0..spec.gpu_count {
                    for j in (i + 1)..spec.gpu_count {
                        g.add_link(&gpu_ids[i], &gpu_ids[j], LinkKind::PCIe, 64.0, 0);
                    }
                }
            }
            TopologyKind::IbChain => {
                for i in 0..(spec.gpu_count - 1) {
                    g.add_link(
                        &gpu_ids[i],
                        &gpu_ids[i + 1],
                        LinkKind::InfiniBand,
                        spec.ib_gbps,
                        1,
                    );
                }
            }
        }
        Ok(Self {
            graph:   Arc::new(RwLock::new(g)),
            gpu_ids,
            config:  spec.clone(),
            seed:    cfg.seed,
        })
    }

    /// Generate a synthetic `MetricSample` for `gpu_id` at logical time `t_ns`.
    ///
    /// Uses a Knuth multiplicative LCG for deterministic, cheap pseudo-noise.
    /// Jitter magnitude is ~2 % of full scale — realistic sensor quantisation noise.
    pub fn sample_at(&mut self, gpu_id: &str, t_ns: u64) -> MetricSample {
        self.seed = self.seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let jitter = (self.seed as f64 / u64::MAX as f64) as f32 * 0.02;
        MetricSample {
            gpu_id:                       gpu_id.to_string(),
            timestamp_ns:                 t_ns,
            ecc_correctable_rate:         0.0 + jitter,
            ecc_uncorrectable_rate:       0.0,
            temperature_celsius:          72.0 + jitter * 5.0,
            nvlink_bandwidth_gbps:        self.config.nvlink_gbps * (0.95 + jitter),
            ib_bandwidth_gbps:            self.config.ib_gbps * (0.98 + jitter),
            sm_utilization:               0.85 + jitter,
            memory_bandwidth_utilization: 0.78 + jitter,
        }
    }
}
