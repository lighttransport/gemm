"""Memory-bounded single-frame execution of the unchanged Diffusers decoder."""
from contextlib import contextmanager


class DiscardFirstFrameCache:
    """Discard cache writes that only a subsequent video frame could use.

    Fail if the decoder ever reads a written slot: that would invalidate
    this optimization. First-frame reads must all observe the initial None.
    No tensor data or arithmetic is changed.
    """
    def __init__(self, size):
        self.size = size
        self.written = set()

    def _check(self, index):
        if not isinstance(index, int) or not 0 <= index < self.size:
            raise IndexError(f"invalid cache index {index}")

    def __getitem__(self, index):
        self._check(index)
        if index in self.written:
            raise RuntimeError("decoder reread a discarded first-frame cache entry")
        return None

    def __setitem__(self, index, value):
        self._check(index)
        if index in self.written:
            raise RuntimeError("decoder rewrote a discarded first-frame cache entry")
        self.written.add(index)


@contextmanager
def discard_single_frame_cache(vae):
    """Keep official decode arithmetic, skip retaining unused temporal caches.

    Use only with a single, untiled frame. A decoder pre-hook substitutes a
    checked write-discard cache for the initially empty feature list. The
    official code still executes all first-chunk branches, including the
    temporal upsamplers' cache-priming branch (unlike feat_cache=None).
    """
    calls = []

    def hook(module, args, kwargs):
        x = args[0]
        cache = kwargs.get("feat_cache")
        if (calls or x.ndim != 5 or x.shape[2] != 1 or
                kwargs.get("first_chunk") is not True or cache is None or
                any(value is not None for value in cache)):
            raise ValueError("cache discard requires one untiled first-frame decoder call")
        replacement = DiscardFirstFrameCache(len(cache))
        calls.append(replacement)
        return args, {**kwargs, "feat_cache": replacement}

    handle = vae.decoder.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        yield calls
        if len(calls) != 1:
            raise RuntimeError("expected exactly one decoder call")
    finally:
        handle.remove()
