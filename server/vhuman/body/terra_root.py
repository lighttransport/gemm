"""Preserve measured source-root motion inside the constrained Terra solve."""
import numpy as np


class SourceRootTerm:
    name = 'vhuman_source_root'
    configured = True

    def __init__(self, targets, weight):
        self.targets = np.array(targets, dtype=float, copy=True)
        self.weight = float(weight)
        if (self.targets.ndim != 2 or self.targets.shape[1] != 3 or not len(self.targets)
                or not np.isfinite(self.targets).all() or not np.isfinite(self.weight) or self.weight <= 0):
            raise ValueError('invalid source-root objective')

    def linearize(self, q, indices, frame):
        if not 0 <= frame < len(self.targets):
            raise ValueError('source-root objective frame exceeds the prepared clock')
        q = np.asarray(q)
        rows = np.eye(len(q))[:3, indices]
        return rows, self.targets[frame] - q[:3]

    def quadratic_terms(self, host, q, last, frame):
        from terra._qp import _row_quadratic_terms
        rows, target = self.linearize(q, host.q_a_indices, frame)
        return _row_quadratic_terms([(self.weight, rows[i], target[i]) for i in range(3)])


class RootTrackingObjective:
    """Extend one solver instance without modifying the registered upstream code."""
    def __init__(self, original, targets, weight):
        self.original = original
        self.root = SourceRootTerm(targets, weight)

    def __getattr__(self, name):
        return getattr(self.original, name)

    def components(self):
        return (*self.original.components(), self.root)

    def active_names(self):
        return (*self.original.active_names(), self.root.name)

    def quadratic_terms(self, host, q, last, frame):
        return (self.original.quadratic_terms(host, q, last, frame)
                + self.root.quadratic_terms(host, q, last, frame))
