"""Analytical MFU model calibrated against NVIDIA NCCL benchmarks (public)."""
from dataclasses import dataclass

# Representative transformer at GPT-3 scale
_PARAMS = 175e9
_SEQ_LEN = 2048
_BATCH = 1
# A100 80GB at bf16
_COMPUTE_TFLOPS = 312e12

# Calibrated from NVIDIA Collective Communications Benchmarks
NEMESIS_BW_GBPS = 600.0   # NVLink 3.0, 8-GPU DGX A100
K8S_BW_GBPS = 25.0        # PCIe 4.0 cross-node effective (shared bus contention)


@dataclass
class BandwidthModel:
    bw_gbps: float
    n_gpus: int

    def __post_init__(self) -> None:
        if self.n_gpus < 1:
            raise ValueError(f"n_gpus must be >= 1, got {self.n_gpus}")
        if self.bw_gbps <= 0:
            raise ValueError(f"bw_gbps must be > 0, got {self.bw_gbps}")

    def all_reduce_ns(self, bytes_: float) -> float:
        """Ring all-reduce latency: 2*(N-1)/N * bytes / bandwidth."""
        bw_bytes_s = self.bw_gbps * 1e9 / 8
        return 2 * (self.n_gpus - 1) / self.n_gpus * bytes_ / bw_bytes_s * 1e9

    def compute_ns(self) -> float:
        """Forward+backward flops / A100 peak TFLOPS."""
        flops = 6 * _PARAMS * _SEQ_LEN * _BATCH
        return flops / _COMPUTE_TFLOPS * 1e9

    def mfu(self, param_dtype_bytes: int = 2) -> float:
        """Model FLOP utilization = compute / (compute + all_reduce)."""
        gradient_bytes = 2 * _PARAMS * param_dtype_bytes
        comp = self.compute_ns()
        comm = self.all_reduce_ns(gradient_bytes)
        return comp / (comp + comm)
