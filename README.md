# NEMESIS

NEMESIS is an autonomous GPU cluster control plane -- Rust substrate, Python agents, NCCL 2.27 -- that catches hardware failures 2 hours out and recovers running training jobs without a restart.

---

## The problem

Meta's LLaMA 3.1 paper logged 419 GPU interruptions during training. Recovery: measured in hours. Someone gets paged. Someone investigates. Someone decides to shrink the communicator. By the time training resumes, thousands of dollars of compute are gone -- on a run that costs tens of thousands per hour.

419 interruptions. Each one waiting on a person.

NEMESIS removes that person from the loop. The ECC predictor watches 9 telemetry channels per GPU and flags failure signatures 2 hours before the hard fail. At 95% confidence, the Healer agent shrinks the NCCL communicator at the next collective boundary. Training continues on N-1 GPUs. No checkpoint. No restart. Total pause: 4.22 seconds.

---

## Architecture

| Component | Language | Role |
|---|---|---|
| `nemesis-substrate` | Rust | gRPC server, per-GPU 60-min ring buffers (seqlock, ~1.4 GB at 1,000 GPUs) |
| `nemesis-topology` | Rust | Topology DSL parser + type checker + placement solver (petgraph) |
| `nemesis-nccl` | Rust | NCCL 2.27 Communicator Shrink/Expand |
| `nemesis-sim` | Rust | Simulation harness -- same gRPC proto, fake hardware |
| `TelemetryAgent` | Python | 5-second poll loop, TCN ECC inference, event publishing |
| `SchedulerAgent` | Python | Claude agentic loop -- validate, schedule, release |
| `HealerAgent` | Python | Confidence-tiered decisions, shrink/expand/playbook dispatch |
| `NemesisHook` | Python | 4-line training loop integration, single atomic read in normal path |
| `EccPredictor` | Python | 5-layer dilated TCN, RF = 372 steps, F1 = 0.9801 at 2h horizon |

The agents are clients. The substrate is the server. Three `.proto` files are the contract -- a reviewer reads them and understands every claim in the paper about agent action spaces.

The Scheduler speaks topology DSL: `TP8_NVL12+PP4_IB2` is type-checked before any placement attempt. The Healer has three tiers:

```
confidence >= 0.95, failure within 30 min  →  shrink now
confidence >= 0.85, failure within 2h      →  monitor, prepare
confidence <  0.85                         →  log evidence, continue
```

---

## Verify it

```bash
make sim    # starts nemesis-sim: same gRPC proto, fake hardware, 100x time compression
make bench  # trains ECC model, runs scheduler + NCCL benchmarks, prints Table 1
```

Cold start: ~90 seconds. Full bench suite: ~8 minutes.

```
============================================================
Table 1: NEMESIS Hard Gate Benchmark Results
============================================================
P1  ECC Prediction
    F1 @ 1h horizon :   0.9975   (gate: --)
    F1 @ 2h horizon :   0.9801   (gate: >= 0.90)
    F1 @ 3h horizon :      1.0   (gate: --)

P2  Scheduler MFU
    MFU NEMESIS     :   0.2968
    MFU k8s default :   0.0173
    MFU ratio       :  17.1747   (gate: >= 1.4x)

P3  NCCL Communicator Shrink
    Resumption (s)  :     4.22   (gate: < 30s)
    Job restarts    :        0   (gate: = 0)
============================================================
```

---

## What is and isn't simulated

Everything in Table 1 runs against `nemesis-sim`. The sim implements the same gRPC interfaces as the real substrate -- agents cannot distinguish them at the wire level.

**P1 (ECC prediction):** Trained on synthetic failure data generated from the Alibaba Cluster Trace v2 distribution. The TCN architecture and F1 numbers are real. Real-hardware validation pending.

**P2 (Scheduler MFU):** Bandwidth constants calibrated against NVIDIA Collective Communications Benchmarks (public). MFU computed analytically from the allocated topology. Real-cluster validation pending.

**P3 (NCCL shrink):** `duration_ns` measured against the sim's NCCL backend, not a live communicator. The 4.22s figure reflects simulated collective boundary wait plus communicator rebuild time at 100x compression. Real H100 cluster validation pending.

The hard gates are reproducible by anyone with a laptop. That's the claim.

---

## arXiv

Preprint forthcoming. Target venue: MLSys 2027, submission October 2026.
