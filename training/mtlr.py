"""Multi-Task Logistic Regression (MTLR) survival loss — from-scratch, no torchmtlr.

Standard Yu-et-al-2011 discrete-time formulation (the same math torchmtlr uses),
implemented here directly so we don't depend on the unpublished/blocked package.

A model produces (K-1) raw "phi" scores per patient. A fixed lower-triangular
matrix G (shape (K-1, K)) accumulates them into K interval logits. The negative
log-likelihood handles censoring via a masked log-sum-exp.

Time axis: K intervals defined by (K-1) ascending boundaries `bins`. Interval k
covers (bins[k-1], bins[k]]; the last interval is (bins[-1], inf).

Risk for c-index = expected event-interval index (higher = earlier event = worse),
which is monotonic in the negated expected survival time — exactly what Harrell's
c-index consumes.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def make_time_bins(times: np.ndarray, events: np.ndarray,
                   num_bins: int = 10) -> torch.Tensor:
    """Quantile bin boundaries over EVENT times (censored excluded from quantiles).

    Returns (num_bins-1,) ascending float tensor — the interior cut points, so the
    model sees `num_bins` intervals.
    """
    ev_times = times[events.astype(bool)]
    if ev_times.size < 2:
        ev_times = times
    qs = np.linspace(0.0, 1.0, num_bins + 1)[1:-1]   # drop 0 and 1 → num_bins-1 cuts
    edges = np.quantile(ev_times, qs)
    edges = np.unique(edges)                          # dedup ties
    return torch.tensor(edges, dtype=torch.float32)


def encode_survival(time: torch.Tensor, event: torch.Tensor,
                    bins: torch.Tensor) -> torch.Tensor:
    """Encode (time, event) into the (B, K) MTLR target, K = len(bins)+1.

    Uncensored at interval k → one-hot at k.
    Censored at interval k   → 1s from k..K-1 (survived past k, may die in any later
    interval).
    """
    B = time.shape[0]
    K = bins.shape[0] + 1
    y = torch.zeros(B, K, device=time.device)
    # interval index = number of boundaries strictly below the time
    idx = torch.searchsorted(bins.to(time.device), time.contiguous())
    idx = idx.clamp(max=K - 1)
    for i in range(B):
        k = int(idx[i])
        if event[i] > 0.5:
            y[i, k] = 1.0
        else:
            y[i, k:] = 1.0
    return y


def _accumulate(phi: torch.Tensor) -> torch.Tensor:
    """phi (B, K-1) raw scores → (B, K) interval logits via lower-triangular G.

    G[i, j] = 1 if j <= i (shape (K-1, K)); a zero column is appended so the last
    interval gets logit 0 (reference). Equivalent to torchmtlr's forward matmul.
    """
    B, Km1 = phi.shape
    K = Km1 + 1
    # cumulative sum from the right: logit_k = sum_{j>=k} phi_j  (k=0..K-1, phi_K-1=0)
    phi_pad = F.pad(phi, (0, 1), value=0.0)                      # (B, K), last col = 0
    # reverse-cumsum so earlier intervals accumulate more terms
    logits = torch.flip(torch.cumsum(torch.flip(phi_pad, dims=[1]), dim=1), dims=[1])
    return logits                                               # (B, K)


def _masked_logsumexp(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """logsumexp over the positions where mask==1, per row. (B,)"""
    neg_inf = torch.finfo(x.dtype).min
    x_masked = x.masked_fill(mask < 0.5, neg_inf)
    return torch.logsumexp(x_masked, dim=1)


def mtlr_neg_log_likelihood(phi: torch.Tensor, target: torch.Tensor,
                            average: bool = True) -> torch.Tensor:
    """MTLR NLL. phi (B, K-1) raw scores; target (B, K) from encode_survival."""
    logits = _accumulate(phi)                                   # (B, K)
    censored = target.sum(dim=1) > 1.5                          # multi-hot → censored
    # numerator: uncensored = logit at the one-hot bin; censored = lse over alive bins
    num = torch.where(
        censored,
        _masked_logsumexp(logits, target),
        (logits * target).sum(dim=1),
    )
    denom = torch.logsumexp(logits, dim=1)                      # partition over K bins
    nll = -(num - denom)
    return nll.mean() if average else nll.sum()


def mtlr_risk(phi: torch.Tensor) -> torch.Tensor:
    """Scalar risk per patient (higher = earlier event = worse). (B,)

    density_k = softmax(interval logits)_k = P(event in interval k);
    expected event interval E[k] is LARGE for late events, so risk = -E[k]
    (negated) so that higher risk = earlier event — the convention Harrell's
    c-index expects (concordant when higher risk ↔ shorter survival).
    """
    logits = _accumulate(phi)
    density = F.softmax(logits, dim=1)                          # (B, K)
    k = torch.arange(density.shape[1], device=density.device, dtype=density.dtype)
    return -(density * k).sum(dim=1)
