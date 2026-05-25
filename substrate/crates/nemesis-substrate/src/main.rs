use nemesis_graph::ClusterGraph;
use nemesis_nccl::service::HealerServiceImpl;
use nemesis_proto::healer::v1::healer_service_server::HealerServiceServer;
use nemesis_proto::telemetry::v1::telemetry_service_server::TelemetryServiceServer;
use nemesis_proto::topology::v1::scheduler_service_server::SchedulerServiceServer;
use nemesis_telemetry::{service::TelemetryServiceImpl, store::TelemetryStore};
use nemesis_topology::service::SchedulerServiceImpl;
use parking_lot::RwLock;
use std::{net::SocketAddr, sync::Arc};
use tonic::transport::Server;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let port: u16 = std::env::args()
        .skip_while(|a| a != "--port")
        .nth(1)
        .and_then(|p| p.parse().ok())
        .unwrap_or(50051);

    let addr: SocketAddr = format!("[::1]:{port}").parse()?;
    tracing::info!("nemesis-substrate listening on {addr}");

    let graph = Arc::new(RwLock::new(ClusterGraph::new()));
    let store = TelemetryStore::new();

    let telemetry_svc = TelemetryServiceImpl::new(store.clone(), graph.clone());
    let scheduler_svc = SchedulerServiceImpl::new(graph.clone());
    let healer_svc    = HealerServiceImpl::new_sim(8, 0);

    Server::builder()
        .add_service(TelemetryServiceServer::new(telemetry_svc))
        .add_service(SchedulerServiceServer::new(scheduler_svc))
        .add_service(HealerServiceServer::new(healer_svc))
        .serve(addr)
        .await?;

    Ok(())
}
