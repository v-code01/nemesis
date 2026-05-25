mod bandwidth;
mod cluster;
mod config;
mod scenario;
mod services {
    pub mod healer;
    pub mod telemetry;
    pub mod topology;
}

use cluster::SimCluster;
use config::SimConfig;
use nemesis_proto::healer::v1::healer_service_server::HealerServiceServer;
use nemesis_proto::telemetry::v1::telemetry_service_server::TelemetryServiceServer;
use nemesis_proto::topology::v1::scheduler_service_server::SchedulerServiceServer;
use nemesis_telemetry::store::TelemetryStore;
use services::healer::HealerServiceImpl;
use services::telemetry::TelemetryServiceImpl;
use services::topology::SchedulerServiceImpl;
use anyhow::Context;
use std::{net::SocketAddr, time::Duration};
use tonic::transport::Server;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let config_path = std::env::args().skip_while(|a| a != "--config").nth(1);
    let cfg: SimConfig = match config_path {
        Some(ref p) => {
            let content = std::fs::read_to_string(p)
                .with_context(|| format!("reading sim config from '{p}'"))?;
            serde_yaml::from_str(&content)
                .with_context(|| format!("parsing sim config YAML from '{p}'"))?
        }
        None => SimConfig::default(),
    };

    anyhow::ensure!(
        cfg.time_scale > 0.0 && cfg.time_scale.is_finite(),
        "time_scale must be a finite positive number, got {}",
        cfg.time_scale
    );

    let addr: SocketAddr = format!("[::1]:{}", cfg.port).parse()?;
    tracing::info!(
        port = cfg.port,
        time_scale = cfg.time_scale,
        seed = cfg.seed,
        "nemesis-sim starting"
    );

    let mut sim_cluster = SimCluster::from_config(&cfg)?;
    let graph = sim_cluster.graph.clone();
    let store = TelemetryStore::new();

    // Background task: emit synthetic metrics every 100 ms, scaled by time_scale.
    // Higher time_scale compresses real time: time_scale=10.0 → 10 ms sleep intervals,
    // making 1 simulated second pass in 100 ms of wall time.
    let store_bg = store.clone();
    let gpu_ids = sim_cluster.gpu_ids.clone();
    let time_scale = cfg.time_scale;
    let metric_task = tokio::spawn(async move {
        let interval_ms = (100.0 / time_scale).max(1.0) as u64;
        let mut t_ns: u64 = 0;
        loop {
            for id in &gpu_ids {
                let sample = sim_cluster.sample_at(id, t_ns);
                store_bg.ingest(sample);
            }
            t_ns += 100_000_000;
            tokio::time::sleep(Duration::from_millis(interval_ms)).await;
        }
    });

    let telemetry_svc = TelemetryServiceImpl::new(store.clone(), graph.clone());
    let scheduler_svc = SchedulerServiceImpl::new(graph.clone());
    let healer_svc    = HealerServiceImpl::new_sim(cfg.cluster.gpu_count as u32, cfg.seed);

    tracing::info!("nemesis-sim listening on {addr}");
    Server::builder()
        .add_service(TelemetryServiceServer::new(telemetry_svc))
        .add_service(SchedulerServiceServer::new(scheduler_svc))
        .add_service(HealerServiceServer::new(healer_svc))
        .serve_with_shutdown(addr, async {
            tokio::signal::ctrl_c().await.ok();
            tracing::info!("shutdown signal received");
        })
        .await?;

    metric_task.abort();

    Ok(())
}
