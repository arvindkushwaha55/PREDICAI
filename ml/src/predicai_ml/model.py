"""PREDICAI — LSTM RUL regressor (Part 4).

Architecture (config-driven, see ml/configs/fd001_lstm.yaml):
    Input (B, 30, F)
    -> LSTM(F -> hidden1) -> Dropout
    -> LSTM(hidden1 -> hidden2) -> Dropout   [last hidden state]
    -> Linear(hidden2 -> head_dim) -> ReLU
    -> Linear(head_dim -> 1)                 [predicted RUL, cycles]

Predictions are clipped at >= 0 at inference time (RUL cannot be negative);
the raw regression output is used during training.
"""
from __future__ import annotations

import torch
from torch import nn


class LSTMRULRegressor(nn.Module):
    def __init__(
        self,
        num_features: int,
        hidden_1: int = 128,
        hidden_2: int = 64,
        head_dim: int = 32,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.lstm1 = nn.LSTM(num_features, hidden_1, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hidden_1, hidden_2, batch_first=True)
        self.drop2 = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_2, head_dim),
            nn.ReLU(),
            nn.Linear(head_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm1(x)          # (B, T, hidden_1)
        out = self.drop1(out)
        out, _ = self.lstm2(out)        # (B, T, hidden_2)
        last = self.drop2(out[:, -1, :])  # last time step
        return self.head(last).squeeze(-1)  # (B,)


def predict_rul(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Inference helper: eval mode, no grad, RUL clipped at 0."""
    model.eval()
    with torch.no_grad():
        return torch.clamp(model(x), min=0.0)
