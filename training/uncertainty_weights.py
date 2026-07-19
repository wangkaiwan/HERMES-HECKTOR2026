"""
Kendall & Gal (CVPR 2018) uncertainty-based multi-task weighting.

Each task k has a learnable log-variance s_k. The effective loss is:
    L = sum_k 0.5 * exp(-s_k) * L_k + 0.5 * s_k

For classification-like losses we drop the 0.5 factor on s_k, following the
common code recipe.
"""
from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn


class UncertaintyWeights(nn.Module):
    def __init__(self, task_names: Iterable[str]) -> None:
        super().__init__()
        self.task_names = list(task_names)
        self.log_vars = nn.Parameter(torch.zeros(len(self.task_names)))

    def forward(self, losses: dict) -> torch.Tensor:
        total = 0.0
        for i, name in enumerate(self.task_names):
            if name not in losses or losses[name] is None:
                continue
            precision = torch.exp(-self.log_vars[i])
            total = total + 0.5 * precision * losses[name] + 0.5 * self.log_vars[i]
        return total

    def state(self) -> dict:
        return {n: float(self.log_vars[i].detach()) for i, n in enumerate(self.task_names)}
