"""Explicit Hunyuan backend selection; repository mode forbids vendor fallback."""
class RepositoryBackend:
    frames = (81,)

    def __init__(self):
        from cuda.hunyuan_video15_native import generate
        self.module = generate
        self.RUNNER = generate.RUNNER
        self.Cancelled = generate.Cancelled

    def load_manifest(self, model, task, preset):
        # The generation path validates every component hash before inference.
        return self.module.model_manifest(model, task, preset, verify=False)[0]

    def generate(self, *, frames=81, **kwargs):
        if frames not in self.frames:
            raise ValueError('repository Hunyuan currently supports 81 frames; 121-frame generation requires the explicit legacy backend')
        result = self.module.generate(**kwargs, gemm='repo', gemm_fallback='error')
        metrics = result.get('metrics', {})
        if (metrics.get('cublas_gemm_calls') != 0 or metrics.get('fallback_gemm_calls') != 0 or
                type(metrics.get('repo_gemm_calls')) is not int or metrics['repo_gemm_calls'] < 1):
            raise RuntimeError('repository Hunyuan reported forbidden vendor GEMM fallback')
        fast = kwargs.get('preset', 'quality') == 'fast12'
        result.update(steps=12 if fast else 50, cfg=1 if fast else 6, flow_shift=7 if fast else 5)
        return result


def select(backend='repo'):
    if backend == 'repo':
        return RepositoryBackend()
    if backend == 'legacy':
        from cuda.hunyuan_video15 import native_generate
        return native_generate
    raise ValueError('video backend must be repo or legacy')
