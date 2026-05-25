// TelemetryServiceImpl: gRPC server implementation for TelemetryService.
//
// RPC surface (from telemetry.proto):
//   IngestMetrics(stream MetricSample) → stream Void   (bidi-streaming)
//   SubscribeEvents(EventFilter)        → stream HardwareEvent (server-streaming)
//   GetClusterState(Void)              → ClusterSnapshot (unary)
//   PublishEvent(HardwareEvent)        → Void (unary)
//
// Design notes:
//   - `TelemetryStore` is cheaply cloneable (Arc-backed); cloned into the
//     ingest_metrics async block so the spawned future owns it without a
//     lifetime dependency on `self`.
//   - `BroadcastStream` converts `broadcast::Receiver` into an async `Stream`.
//     Lagged receivers receive `BroadcastStreamRecvError::Lagged(n)`, which we
//     log at WARN and skip — we never back-pressure the publisher.
//   - `ClusterGraph::to_proto()` serialises the topology on demand; the
//     write-lock on `graph` is released before the gRPC response is sent.

use crate::store::TelemetryStore;
use nemesis_graph::ClusterGraph;
use nemesis_proto::telemetry::v1::{
    telemetry_service_server::TelemetryService,
    ClusterSnapshot, EventFilter, HardwareEvent, MetricSample, Void,
};
use parking_lot::RwLock;
use std::pin::Pin;
use std::sync::Arc;
use tokio_stream::wrappers::BroadcastStream;
use tokio_stream::StreamExt as _;
use tonic::{Request, Response, Status, Streaming};

/// Pinned, heap-allocated `Stream` — the return type for both streaming RPCs.
type BoxStream<T> = Pin<Box<dyn tokio_stream::Stream<Item = Result<T, Status>> + Send + 'static>>;

/// Concrete gRPC service implementation.
pub struct TelemetryServiceImpl {
    pub store: TelemetryStore,
    pub graph: Arc<RwLock<ClusterGraph>>,
}

impl TelemetryServiceImpl {
    pub fn new(store: TelemetryStore, graph: Arc<RwLock<ClusterGraph>>) -> Self {
        Self { store, graph }
    }
}

#[tonic::async_trait]
impl TelemetryService for TelemetryServiceImpl {
    // --- associated stream types required by the generated trait ---

    type IngestMetricsStream    = BoxStream<Void>;
    type SubscribeEventsStream  = BoxStream<HardwareEvent>;

    // --- RPC implementations ---

    /// Bidirectional streaming ingest.
    ///
    /// Each received `MetricSample` is stored in the corresponding GPU ring
    /// and acknowledged with a `Void` response.  The stream terminates when
    /// the client closes its side; errors from the transport layer are
    /// forwarded verbatim to the response stream.
    async fn ingest_metrics(
        &self,
        request: Request<Streaming<MetricSample>>,
    ) -> Result<Response<Self::IngestMetricsStream>, Status> {
        // Clone the store (O(1) Arc clone) so the async block is 'static.
        let store = self.store.clone();
        let mut stream = request.into_inner();

        let output = async_stream::stream! {
            while let Some(result) = stream.next().await {
                match result {
                    Ok(sample) => {
                        store.ingest(sample);
                        yield Ok(Void {});
                    }
                    Err(status) => yield Err(status),
                }
            }
        };

        Ok(Response::new(Box::pin(output)))
    }

    /// Server-streaming event subscription.
    ///
    /// Subscribes to the broadcast bus and forwards every `HardwareEvent`
    /// until the client disconnects.  Events dropped due to receiver lag are
    /// logged at WARN level and skipped — downstream clients should re-request
    /// cluster state if they detect a gap.
    async fn subscribe_events(
        &self,
        _request: Request<EventFilter>,
    ) -> Result<Response<Self::SubscribeEventsStream>, Status> {
        let rx = self.store.subscribe();

        let stream = BroadcastStream::new(rx).filter_map(|result| {
            match result {
                Ok(event) => Some(Ok(event)),
                Err(tokio_stream::wrappers::errors::BroadcastStreamRecvError::Lagged(n)) => {
                    tracing::warn!(
                        dropped = n,
                        "telemetry event subscriber lagged; {n} events dropped"
                    );
                    // Skip the gap; do not terminate the stream.
                    None
                }
            }
        });

        Ok(Response::new(Box::pin(stream)))
    }

    /// Unary snapshot of the full cluster state.
    ///
    /// Serialises the topology graph and fetches the latest sample per GPU
    /// under a single read-lock acquisition each.  The `taken_ns` field is
    /// stamped with the current wall-clock time.
    async fn get_cluster_state(
        &self,
        _request: Request<Void>,
    ) -> Result<Response<ClusterSnapshot>, Status> {
        // Serialise the topology; lock is dropped at the end of this block.
        let topology = self.graph.read().to_proto();

        // Collect the single most-recent sample for each known GPU.
        let latest: Vec<MetricSample> = self
            .store
            .gpu_ids()
            .into_iter()
            .filter_map(|id| self.store.window(&id, 1).into_iter().next())
            .collect();

        let taken_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;

        Ok(Response::new(ClusterSnapshot {
            topology: Some(topology),
            latest,
            taken_ns,
        }))
    }

    /// Unary publish: broadcast a `HardwareEvent` to all active subscribers.
    ///
    /// Returns immediately.  If there are no active subscribers the event is
    /// silently dropped (best-effort bus semantics).
    async fn publish_event(
        &self,
        request: Request<HardwareEvent>,
    ) -> Result<Response<Void>, Status> {
        self.store.publish(request.into_inner());
        Ok(Response::new(Void {}))
    }
}
