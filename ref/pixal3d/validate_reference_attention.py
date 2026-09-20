"""Validate the bounded Blackwell NAF fallback against a direct reference."""
import torch

from run_reference_mv import _DeferredNaf, _chunked_na2d, _deferred_pixels


def direct(q, k, v, kernel_size, dilation, scale):
    height, width = q.shape[1:3]
    output = torch.empty((*q.shape[:-1], v.shape[-1]), device=q.device, dtype=v.dtype)
    radius = kernel_size // 2
    span = (kernel_size - 1) * dilation
    for row in range(height):
        row0 = min(max(row - radius * dilation, 0), height - 1 - span)
        for col in range(width):
            col0 = min(max(col - radius * dilation, 0), width - 1 - span)
            keys = torch.stack([k[:, row0 + iy * dilation, col0 + ix * dilation]
                                for iy in range(kernel_size) for ix in range(kernel_size)], dim=-2)
            values = torch.stack([v[:, row0 + iy * dilation, col0 + ix * dilation]
                                  for iy in range(kernel_size) for ix in range(kernel_size)], dim=-2)
            weights = torch.softmax((q[:, row, col, :, None] * keys).sum(-1) * scale, dim=-1)
            output[:, row, col] = (weights[..., None] * values).sum(-2)
    return output


def main():
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    q = torch.randn(1, 9, 10, 2, 4, device=device, dtype=torch.float32)
    k = torch.randn_like(q)
    v = torch.randn(1, 9, 10, 2, 7, device=device, dtype=torch.float32)
    scale = q.shape[-1] ** -0.5
    actual = _chunked_na2d(q, k, v, 3, 2, scale)
    expected = direct(q, k, v, 3, 2, scale)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    direct_error = (actual - expected).abs().max().item()

    deferred = object.__new__(_DeferredNaf)
    deferred.heads = 2
    deferred.kernel_size = (3, 3)
    deferred.scale = scale
    deferred.height, deferred.width = 9, 10
    deferred.query = q.permute(0, 3, 4, 1, 2).flatten(1, 2)
    source_k = torch.randn(1, 8, 3, 5, device=device)
    source_v = torch.randn(1, 14, 3, 5, device=device)
    deferred.key, deferred.value = source_k, source_v
    dense_k = torch.nn.functional.interpolate(source_k, (9, 10), mode="nearest-exact")
    dense_v = torch.nn.functional.interpolate(source_v, (9, 10), mode="nearest-exact")
    dense_k = dense_k.reshape(1, 2, 4, 9, 10).permute(0, 3, 4, 1, 2)
    dense_v = dense_v.reshape(1, 2, 7, 9, 10).permute(0, 3, 4, 1, 2)
    dense_expected = _chunked_na2d(q, dense_k, dense_v, 3, (3, 2), scale)
    rows, cols = torch.meshgrid(torch.arange(9, device=device),
                                torch.arange(10, device=device), indexing="ij")
    sparse_actual = _deferred_pixels(deferred, rows.flatten()[None], cols.flatten()[None])
    sparse_actual = sparse_actual.reshape_as(dense_expected)
    torch.testing.assert_close(sparse_actual, dense_expected, rtol=1e-6, atol=1e-6)
    print({"device": device, "chunked_max_abs": direct_error,
           "deferred_max_abs": (sparse_actual - dense_expected).abs().max().item()})


if __name__ == "__main__":
    main()
