use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct SimConfig {
    pub seed:       u64,
    pub time_scale: f64,
    pub port:       u16,
    pub cluster:    ClusterSpec,
}

#[derive(Debug, Deserialize, Clone)]
pub struct ClusterSpec {
    pub gpu_count:   usize,
    pub topology:    TopologyKind,
    pub nvlink_gbps: f32,
    pub ib_gbps:     f32,
}

#[derive(Debug, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum TopologyKind {
    NvlinkFull,
    PcieCrossNode,
    IbChain,
}

impl Default for SimConfig {
    fn default() -> Self {
        Self {
            seed: 42,
            time_scale: 1.0,
            port: 50051,
            cluster: ClusterSpec {
                gpu_count:   8,
                topology:    TopologyKind::NvlinkFull,
                nvlink_gbps: 600.0,
                ib_gbps:     200.0,
            },
        }
    }
}
