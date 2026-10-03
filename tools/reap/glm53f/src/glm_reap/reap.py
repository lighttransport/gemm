from __future__ import annotations

import numpy as np


class Saliency:
    def __init__(self, experts):
        self.count = np.zeros(experts, dtype=np.int64)
        self.weighted_norm = np.zeros(experts, dtype=np.float64)

    def update(self, expert, output, gates, valid=None):
        import torch
        norms = torch.linalg.vector_norm(output.detach().float(), dim=-1)
        if valid is not None:
            norms, gates = norms[valid], gates[valid]
        self.count[expert] += len(norms)
        self.weighted_norm[expert] += float((norms * gates.detach().float()).double().sum().cpu())

    def result(self):
        scores = np.divide(self.weighted_norm, self.count, out=np.zeros_like(self.weighted_norm), where=self.count > 0)
        return {"count": self.count.tolist(), "weighted_norm_sum": self.weighted_norm.tolist(), "scores": scores.tolist()}


def select(scores, keep):
    values = np.asarray(scores, dtype=np.float64)
    if not np.isfinite(values).all() or not 1 <= keep <= len(values):
        raise ValueError("Invalid REAP scores or keep count")
    ranked = np.lexsort((np.arange(len(values)), -values))[:keep]
    return sorted(int(i) for i in ranked)


def route(hidden, weight, correction, top_k=8, scale=2.5, groups=1, top_groups=1, normalize=True):
    import torch
    logits = torch.nn.functional.linear(hidden.float(), weight.float())
    scores = logits.sigmoid()
    choice = scores + correction.float()
    if groups != 1:
        if weight.shape[0] % groups:
            raise ValueError("Expert count must be divisible by group count")
        grouped = choice.reshape(-1, groups, weight.shape[0] // groups)
        selected = grouped.topk(2, dim=-1).values.sum(-1).topk(top_groups, dim=-1).indices
        mask = torch.zeros_like(grouped[:, :, 0], dtype=torch.bool).scatter_(1, selected, True)
        choice = choice.masked_fill(~mask[:, :, None].expand_as(grouped).reshape_as(choice), -torch.inf)
    indices = choice.topk(top_k, dim=-1, sorted=False).indices
    gates = scores.gather(1, indices)
    if normalize:
        gates = gates / (gates.sum(-1, keepdim=True) + 1e-20)
    return indices, gates * scale
