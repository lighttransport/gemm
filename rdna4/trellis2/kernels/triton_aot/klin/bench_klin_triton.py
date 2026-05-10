#!/usr/bin/env python3
"""POC: Triton BF16 matmul + FP32 bias for tex_dec klin shapes.

Y[M,N] f32 = (X[M,K] bf16 @ W[N,K] bf16.T) + bias[N] f32

Worst-offender shape: M=8452 K=2048 N=512 (klin_dn stage 1) — hipBLASLt picks
algo 73680 here, ~60 ms total across the 16 stage-1 calls.
"""
import torch, triton, triton.language as tl, time, sys

@triton.jit
def klin_bf16_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    M, N, K,
    sxm: tl.constexpr, sxk: tl.constexpr,
    swn: tl.constexpr, swk: tl.constexpr,
    sym: tl.constexpr, syn: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)

    X_block = X_ptr + rm[:, None] * sxm + rk[None, :] * sxk
    W_block = W_ptr + rn[:, None] * swn + rk[None, :] * swk

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    mask_m = rm < M
    mask_n = rn < N
    for k0 in range(0, K, BK):
        mask_k = (k0 + rk) < K
        x = tl.load(X_block, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        w = tl.load(W_block, mask=mask_n[:, None] & mask_k[None, :], other=0.0)
        acc += tl.dot(x, tl.trans(w))
        X_block += BK * sxk
        W_block += BK * swk

    bias = tl.load(B_ptr + rn, mask=mask_n, other=0.0)
    acc = acc + bias[None, :]
    Y = Y_ptr + rm[:, None] * sym + rn[None, :] * syn
    tl.store(Y, acc, mask=mask_m[:, None] & mask_n[None, :])


def run(M, K, N, BM, BN, BK, num_warps, num_stages, iters=100):
    torch.manual_seed(0)
    dev = "cuda"
    X = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    W = torch.randn(N, K, device=dev, dtype=torch.bfloat16)
    B = torch.randn(N, device=dev, dtype=torch.float32)
    Y = torch.empty(M, N, device=dev, dtype=torch.float32)

    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    klin_bf16_kernel[grid](
        X, W, B, Y, M, N, K,
        X.stride(0), X.stride(1),
        W.stride(0), W.stride(1),
        Y.stride(0), Y.stride(1),
        BM=BM, BN=BN, BK=BK,
        num_warps=num_warps, num_stages=num_stages,
    )
    torch.cuda.synchronize()

    # correctness vs torch
    ref = (X.float() @ W.float().T) + B
    err = (Y - ref).abs().max().item()
    rel = err / ref.abs().max().item()
    if rel > 1e-2:
        return None

    # bench
    for _ in range(10):
        klin_bf16_kernel[grid](
            X, W, B, Y, M, N, K,
            X.stride(0), X.stride(1),
            W.stride(0), W.stride(1),
            Y.stride(0), Y.stride(1),
            BM=BM, BN=BN, BK=BK,
            num_warps=num_warps, num_stages=num_stages,
        )
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        klin_bf16_kernel[grid](
            X, W, B, Y, M, N, K,
            X.stride(0), X.stride(1),
            W.stride(0), W.stride(1),
            Y.stride(0), Y.stride(1),
            BM=BM, BN=BN, BK=BK,
            num_warps=num_warps, num_stages=num_stages,
        )
    torch.cuda.synchronize()
    ms = (time.time() - t0) * 1000 / iters
    return ms


def main():
    # All klin shapes from tex_dec (M, K, N). M chunked at 16384 max.
    shapes = [
        # (M, K, N, label)
        (1905,  1024, 4096, "stage0 klin_up"),
        (1905,  4096, 1024, "stage0 klin_dn"),
        (8452,   512, 2048, "stage1 klin_up"),
        (8452,  2048,  512, "stage1 klin_dn"),
        (16384,  256, 1024, "stage2 klin_up chunk"),
        (16384, 1024,  256, "stage2 klin_dn chunk"),
        (16384,  128,  512, "stage3 klin_up chunk"),
        (16384,  512,  128, "stage3 klin_dn chunk"),
    ]
    configs = [
        # (BM, BN, BK, nw, ns)
        (64, 128, 64, 4, 2),
        (128, 128, 32, 4, 2),
        (128, 128, 32, 8, 2),
        (128, 64, 64, 4, 2),
        (64, 64, 64, 4, 2),
        (64, 256, 32, 8, 2),
        (256, 64, 32, 4, 2),
    ]
    print(f"{'shape':38s} {'best_cfg':28s} {'ms':>7s} {'TF/s':>7s}")
    for M, K, N, label in shapes:
        best_cfg, best_ms = None, 1e9
        for c in configs:
            try:
                ms = run(M, K, N, *c, iters=50)
            except Exception as e:
                print(f"  {label} {c} FAILED: {e}")
                continue
            if ms and ms < best_ms:
                best_cfg, best_ms = c, ms
        flops = 2 * M * K * N
        tflops = flops / (best_ms * 1e9) if best_ms < 1e9 else 0
        print(f"{label+f' M={M} K={K} N={N}':38s} {str(best_cfg):28s} {best_ms:7.3f} {tflops:7.1f}")


if __name__ == "__main__":
    main()
