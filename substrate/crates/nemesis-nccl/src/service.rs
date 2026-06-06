//! gRPC service implementation for HealerService.
//!
//! `HealerServiceImpl` wires an `Arc<dyn NcclBackend>` into the generated tonic
//! server trait.  Job-to-communicator mapping and per-job world sizes are tracked
//! in `RwLock`-protected `HashMap`s so concurrent RPCs from different jobs do not
//! contend on a single mutex.
//!
//! # Playbook execution
//!
//! `execute_playbook` loads YAML playbook definitions from the directory specified
//! by `NEMESIS_PLAYBOOK_DIR` (or supplied at construction time).  Each playbook
//! defines an ordered list of steps.  Steps with `action: shrink_communicator`
//! call through to the NCCL backend directly; all other steps are executed as
//! structured log events and recorded in `actions_taken`.
//!
//! # Concurrency model
//!
//!   - `register_job`: write locks on `job_comms` and `world_sizes`.
//!   - `shrink_communicator` / `expand_communicator`: no locks; backend is `Send + Sync`.
//!   - `execute_playbook`: reads playbook from filesystem, then dispatches steps;
//!     shrink steps acquire no additional locks beyond the backend's own serialisation.

use crate::backend::NcclBackend;
use crate::sim::NcclSim;
use nemesis_proto::healer::v1::{
    healer_service_server::HealerService, ExpandRequest, ExpandResult, PlaybookLibrary,
    PlaybookRequest, PlaybookResult, RegisterJobRequest, RegisterJobResponse, ShrinkRequest,
    ShrinkResult,
};
use nemesis_proto::telemetry::v1::Void;
use parking_lot::RwLock;
use serde::Deserialize;
use std::{
    collections::HashMap,
    path::PathBuf,
    sync::Arc,
};
use tonic::{Request, Response, Status};
use uuid::Uuid;

// ---------------------------------------------------------------------------
// Playbook YAML schema
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct PlaybookDef {
    name: String,
    description: String,
    #[serde(default)]
    trigger: String,
    steps: Vec<PlaybookStep>,
}

#[derive(Debug, Deserialize)]
struct PlaybookStep {
    action: String,
    #[serde(default)]
    description: String,
    #[serde(default)]
    params: HashMap<String, String>,
}

// ---------------------------------------------------------------------------
// Service
// ---------------------------------------------------------------------------

/// Concrete implementation of the `HealerService` gRPC service.
pub struct HealerServiceImpl {
    backend: Arc<dyn NcclBackend>,
    /// Maps job_id → communicator_id (UUID string).
    job_comms: Arc<RwLock<HashMap<String, String>>>,
    /// Maps job_id → world_size at registration time.
    world_sizes: Arc<RwLock<HashMap<String, u32>>>,
    /// Directory containing YAML playbook definitions.
    playbook_dir: Option<PathBuf>,
}

impl HealerServiceImpl {
    /// Construct a `HealerServiceImpl` backed by `NcclSim`.
    ///
    /// `playbook_dir` is read from `NEMESIS_PLAYBOOK_DIR` if not supplied.
    pub fn new_sim(default_world_size: u32, seed: u64) -> Self {
        let playbook_dir = std::env::var("NEMESIS_PLAYBOOK_DIR")
            .ok()
            .map(PathBuf::from)
            .filter(|p| p.is_dir());

        Self {
            backend: Arc::new(NcclSim::new(default_world_size, seed)),
            job_comms: Arc::new(RwLock::new(HashMap::new())),
            world_sizes: Arc::new(RwLock::new(HashMap::new())),
            playbook_dir,
        }
    }

    /// Load and parse a single playbook YAML file by name.
    fn load_playbook(&self, name: &str) -> anyhow::Result<PlaybookDef> {
        let dir = self
            .playbook_dir
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("no playbook directory configured (set NEMESIS_PLAYBOOK_DIR)"))?;

        let path = dir.join(format!("{name}.yaml"));
        let content = std::fs::read_to_string(&path)
            .map_err(|e| anyhow::anyhow!("read {}: {e}", path.display()))?;
        let def: PlaybookDef = serde_yaml::from_str(&content)
            .map_err(|e| anyhow::anyhow!("parse {}: {e}", path.display()))?;
        Ok(def)
    }

    /// Expand `{{key}}` template tokens in `s` using `params`.
    fn expand_template(s: &str, params: &HashMap<String, String>) -> String {
        let mut out = s.to_string();
        for (k, v) in params {
            out = out.replace(&format!("{{{{{k}}}}}"), v);
        }
        out
    }

    /// Parse a comma-separated rank list like `"3,4"` → `vec![3, 4]`.
    fn parse_ranks(s: &str) -> Vec<u32> {
        s.split(',')
            .filter_map(|p| p.trim().parse::<u32>().ok())
            .collect()
    }
}

// ---------------------------------------------------------------------------
// gRPC trait impl
// ---------------------------------------------------------------------------

#[tonic::async_trait]
impl HealerService for HealerServiceImpl {
    /// Register a training job and return a new communicator UUID.
    async fn register_job(
        &self,
        req: Request<RegisterJobRequest>,
    ) -> Result<Response<RegisterJobResponse>, Status> {
        let r = req.into_inner();
        let comm_id = Uuid::new_v4().to_string();
        {
            let mut comms = self.job_comms.write();
            if let Some(existing) = comms.get(&r.job_id) {
                tracing::warn!(
                    job_id = %r.job_id,
                    old_comm = %existing,
                    new_comm = %comm_id,
                    "duplicate register_job: overwriting existing communicator"
                );
            }
            comms.insert(r.job_id.clone(), comm_id.clone());
        }
        self.world_sizes.write().insert(r.job_id, r.world_size);
        Ok(Response::new(RegisterJobResponse {
            communicator_id: comm_id,
        }))
    }

    /// Shrink the NCCL communicator by removing `exclude_ranks`.
    async fn shrink_communicator(
        &self,
        req: Request<ShrinkRequest>,
    ) -> Result<Response<ShrinkResult>, Status> {
        let r = req.into_inner();
        tracing::info!(
            comm_id = %r.communicator_id,
            excluded = r.exclude_ranks.len(),
            "shrink requested"
        );
        let metrics = self
            .backend
            .shrink(&r)
            .await
            .map_err(|e| Status::internal(e.to_string()))?;
        Ok(Response::new(ShrinkResult {
            success: metrics.success,
            duration_ns: metrics.duration_ns,
            active_rank_count: metrics.active_rank_count,
            error: metrics.error,
        }))
    }

    /// Expand the NCCL communicator by admitting `new_gpu_ids`.
    async fn expand_communicator(
        &self,
        req: Request<ExpandRequest>,
    ) -> Result<Response<ExpandResult>, Status> {
        let r = req.into_inner();
        tracing::info!(
            comm_id = %r.communicator_id,
            added = r.new_gpu_ids.len(),
            "expand requested"
        );
        let metrics = self
            .backend
            .expand(&r)
            .await
            .map_err(|e| Status::internal(e.to_string()))?;
        Ok(Response::new(ExpandResult {
            success: metrics.success,
            duration_ns: metrics.duration_ns,
            active_rank_count: metrics.active_rank_count,
            error: metrics.error,
        }))
    }

    /// Execute a named playbook, dispatching each step in order.
    ///
    /// Steps with `action: shrink_communicator` call through to the NCCL backend
    /// with the `exclude_ranks` and `comm_id` resolved from `req.parameters`.
    /// All other steps are recorded as structured log events and added to
    /// `actions_taken`.
    ///
    /// Returns immediately with `success: false` if the playbook YAML cannot be
    /// found or parsed; individual step failures are logged but do not abort
    /// remaining steps (best-effort execution).
    async fn execute_playbook(
        &self,
        req: Request<PlaybookRequest>,
    ) -> Result<Response<PlaybookResult>, Status> {
        let r = req.into_inner();
        let t0 = std::time::Instant::now();

        let playbook = match self.load_playbook(&r.name) {
            Ok(pb) => pb,
            Err(e) => {
                tracing::warn!(playbook = %r.name, error = %e, "playbook load failed");
                return Ok(Response::new(PlaybookResult {
                    success: false,
                    duration_ns: t0.elapsed().as_nanos() as u64,
                    actions_taken: vec![],
                    error: e.to_string(),
                }));
            }
        };

        let mut actions_taken: Vec<String> = Vec::new();

        for step in &playbook.steps {
            // Expand {{template}} tokens in all param values using request parameters.
            let resolved: HashMap<String, String> = step
                .params
                .iter()
                .map(|(k, v)| (k.clone(), Self::expand_template(v, &r.parameters)))
                .collect();

            match step.action.as_str() {
                "shrink_communicator" => {
                    // Resolve communicator_id from request parameters; fall back to first
                    // registered communicator for the job if not supplied explicitly.
                    let comm_id = resolved
                        .get("comm_id")
                        .or_else(|| r.parameters.get("comm_id"))
                        .cloned()
                        .unwrap_or_else(|| {
                            r.parameters
                                .get("job_id")
                                .and_then(|jid| self.job_comms.read().get(jid).cloned())
                                .unwrap_or_default()
                        });

                    let job_id = r
                        .parameters
                        .get("job_id")
                        .cloned()
                        .unwrap_or_default();

                    let exclude_ranks = resolved
                        .get("exclude_ranks")
                        .map(|s| Self::parse_ranks(s))
                        .unwrap_or_default();

                    let shrink_req = ShrinkRequest {
                        communicator_id: comm_id.clone(),
                        job_id: job_id.clone(),
                        exclude_ranks,
                    };

                    match self.backend.shrink(&shrink_req).await {
                        Ok(m) if m.success => {
                            let action = format!(
                                "shrink_communicator comm={comm_id} \
                                 exclude={} active_after={}",
                                shrink_req.exclude_ranks
                                    .iter()
                                    .map(|r| r.to_string())
                                    .collect::<Vec<_>>()
                                    .join(","),
                                m.active_rank_count
                            );
                            tracing::info!(playbook = %r.name, action = %action);
                            actions_taken.push(action);
                        }
                        Ok(m) => {
                            tracing::warn!(
                                playbook = %r.name,
                                comm_id = %comm_id,
                                error = %m.error,
                                "shrink_communicator step failed"
                            );
                            actions_taken.push(format!("shrink_communicator FAILED: {}", m.error));
                        }
                        Err(e) => {
                            tracing::error!(playbook = %r.name, error = %e, "shrink_communicator error");
                            actions_taken.push(format!("shrink_communicator ERROR: {e}"));
                        }
                    }
                }

                "drain_gpu" => {
                    let gpu_id = resolved
                        .get("gpu_id")
                        .or_else(|| r.parameters.get("gpu_id"))
                        .cloned()
                        .unwrap_or_default();
                    let action = format!("drain_gpu gpu={gpu_id}: {}", step.description);
                    tracing::info!(playbook = %r.name, gpu_id = %gpu_id, "drain_gpu");
                    actions_taken.push(action);
                }

                "checkpoint_jobs" => {
                    let job_ids = resolved
                        .get("job_ids")
                        .or_else(|| r.parameters.get("active_job_ids"))
                        .cloned()
                        .unwrap_or_default();
                    let action = format!("checkpoint_jobs jobs=[{job_ids}]: {}", step.description);
                    tracing::info!(playbook = %r.name, job_ids = %job_ids, "checkpoint_jobs");
                    actions_taken.push(action);
                }

                "notify" => {
                    let event_kind = resolved
                        .get("event_kind")
                        .cloned()
                        .unwrap_or_else(|| "UNSPECIFIED".to_string());
                    let gpu_id = resolved
                        .get("gpu_id")
                        .or_else(|| r.parameters.get("gpu_id"))
                        .cloned()
                        .unwrap_or_default();
                    let action = format!("notify event={event_kind} gpu={gpu_id}");
                    tracing::info!(playbook = %r.name, event_kind = %event_kind, gpu_id = %gpu_id, "notify");
                    actions_taken.push(action);
                }

                other => {
                    let action = format!("{other}: {}", step.description);
                    tracing::info!(playbook = %r.name, action = %action, "unknown step executed as log");
                    actions_taken.push(action);
                }
            }
        }

        Ok(Response::new(PlaybookResult {
            success: true,
            duration_ns: t0.elapsed().as_nanos() as u64,
            actions_taken,
            error: String::new(),
        }))
    }

    /// List all available playbooks from the configured playbook directory.
    async fn list_playbooks(
        &self,
        _req: Request<Void>,
    ) -> Result<Response<PlaybookLibrary>, Status> {
        let Some(dir) = self.playbook_dir.as_deref() else {
            return Ok(Response::new(PlaybookLibrary { playbooks: vec![] }));
        };

        let entries = match std::fs::read_dir(dir) {
            Ok(e) => e,
            Err(err) => {
                tracing::warn!(dir = %dir.display(), error = %err, "cannot read playbook dir");
                return Ok(Response::new(PlaybookLibrary { playbooks: vec![] }));
            }
        };

        let mut playbooks = Vec::new();
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("yaml") {
                continue;
            }
            let content = match std::fs::read_to_string(&path) {
                Ok(c) => c,
                Err(_) => continue,
            };
            if let Ok(def) = serde_yaml::from_str::<PlaybookDef>(&content) {
                playbooks.push(nemesis_proto::healer::v1::Playbook {
                    name: def.name,
                    description: def.description,
                    trigger: def.trigger,
                });
            }
        }
        playbooks.sort_by(|a, b| a.name.cmp(&b.name));

        Ok(Response::new(PlaybookLibrary { playbooks }))
    }
}
