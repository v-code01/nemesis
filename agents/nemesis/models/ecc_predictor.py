from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

FEATURE_NAMES = [
    "ecc_corr_rate",
    "ecc_uncorr_rate",
    "temp",
    "sm_util",
    "mem_bw",
    "nvlink_bw",
    "ib_bw",
    "ecc_corr_delta",
    "ecc_uncorr_delta",
]
N_FEATURES = 9
N_HORIZONS = 3  # 1h, 2h, 3h
SEQ_LEN = 600
THRESHOLD = 0.85


class TemporalBlock(nn.Module):
    """
    Dilated causal Conv1d block with residual connection.

    Inputs:  (batch, in_channels, seq_len)
    Outputs: (batch, out_channels, seq_len)

    Causality invariant: output at time t depends only on inputs at times ≤ t.
    Sequence-length invariant: output seq_len == input seq_len (enforced by _chomp).

    Complexity: O(seq_len * kernel_size) per conv layer.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
    ) -> None:
        super().__init__()
        # Padding (kernel_size - 1) * dilation pads the left and right equally
        # for Conv1d with 'same'-style padding. _chomp removes the right-side
        # padding to restore causality and preserve seq_len.
        self._padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=self._padding,
            dilation=dilation,
        )
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            padding=self._padding,
            dilation=dilation,
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        # 1x1 projection to align channel dims for residual addition
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else None
        )

    def _chomp(self, x: torch.Tensor) -> torch.Tensor:
        """
        Remove the future-leaking right-side padding.

        Conv1d with padding=p adds p zeros to both sides, producing a tensor
        of length seq_len + p. Slicing [:, :, :-p] restores seq_len and ensures
        each output position can only attend to past/present inputs (causal).
        """
        return x[:, :, : -self._padding] if self._padding > 0 else x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self._chomp(self.conv1(x)))
        out = self.dropout(out)
        out = self.relu(self._chomp(self.conv2(out)))
        out = self.dropout(out)
        res = x if self.downsample is None else self.downsample(x)
        # Residual add + ReLU: stabilises gradients; matches TCN paper (Bai 2018)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """
    Stack of TemporalBlocks with exponentially increasing dilation.

    Receptive field: sum over layers of (kernel_size - 1) * dilation.
    With kernel_size=7, channels=[64,128,128,64]: RF = 6*(1+2+4+8) = 90 steps.

    Inputs:  (batch, num_inputs, seq_len)
    Outputs: (batch, channels[-1], seq_len)
    """

    def __init__(
        self,
        num_inputs: int,
        channels: list[int],
        kernel_size: int = 7,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for i, out_ch in enumerate(channels):
            in_ch = num_inputs if i == 0 else channels[i - 1]
            # Dilation doubles per layer: captures multi-scale temporal patterns
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, dilation=2**i))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class EccPredictor(nn.Module):
    """
    Temporal Convolutional Network predicting ECC failure probability at 1h/2h/3h horizons.

    Architecture:
        input (batch, 600, 9)
        → transpose → (batch, 9, 600)
        → TCN([64,128,128,64], kernel_size=7) → (batch, 64, 600)
        → last timestep → (batch, 64)
        → Linear(64, 3) → Sigmoid → (batch, 3)  ∈ [0,1]

    The last-timestep read-out is causal: no future information leaks into inference.
    """

    def __init__(self) -> None:
        super().__init__()
        self.tcn = TemporalConvNet(N_FEATURES, channels=[64, 128, 128, 64], kernel_size=7)
        self.head = nn.Linear(64, N_HORIZONS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, n_features) — time-major input matching numpy convention

        Returns:
            probs: (batch, 3) — sigmoid probabilities for 1h/2h/3h horizons
        """
        # Transpose to channel-first for Conv1d: (batch, features, seq_len)
        out = self.tcn(x.transpose(1, 2))
        # Read out only the last (most recent) timestep: (batch, 64)
        return torch.sigmoid(self.head(out[:, :, -1]))

    def infer(self, window: np.ndarray) -> tuple[float, float, float]:
        """
        Run inference on a single telemetry window.

        Args:
            window: (SEQ_LEN=600, N_FEATURES=9) float32 numpy array

        Returns:
            (p_1h, p_2h, p_3h) — scalar failure probabilities in [0, 1]
        """
        self.eval()
        with torch.no_grad():
            probs = self.forward(
                torch.from_numpy(window).float().unsqueeze(0)
            ).squeeze(0)
        return float(probs[0]), float(probs[1]), float(probs[2])

    def explain(self, window: np.ndarray) -> dict[str, float]:
        """
        Compute per-feature importance at the 2h horizon via input-gradient magnitude.

        Method: |∂p_2h / ∂x|, averaged over the time axis, then L1-normalised so
        importances sum to 1.0. Suitable for real-time dashboard display.

        Args:
            window: (SEQ_LEN, N_FEATURES) float32 numpy array

        Returns:
            dict mapping each feature name → normalised importance in [0, 1]
            (values sum to 1.0 within 1e-4 tolerance)
        """
        self.eval()
        x = torch.from_numpy(window).float().unsqueeze(0).requires_grad_(True)
        # Backprop through p_2h (index 1) for single sample
        self.forward(x)[0, 1].backward()
        # Mean absolute gradient over time dimension → (N_FEATURES,)
        grad = x.grad.squeeze(0).abs().mean(dim=0)
        # L1-normalise; epsilon prevents division-by-zero on zero-gradient inputs
        total = grad.sum().item() + 1e-9
        return {name: float(grad[i]) / total for i, name in enumerate(FEATURE_NAMES)}

    def save(self, path: str | Path) -> None:
        """Persist model weights to disk."""
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str | Path) -> "EccPredictor":
        """Restore model from a weights file saved by save()."""
        model = cls()
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        return model
