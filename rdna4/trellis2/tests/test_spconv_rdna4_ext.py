"""Standalone correctness test for spconv_rdna4_ext.

Generates random sparse coords + feats + weight, computes a CPU reference
(gather-then-matmul over the 3x3x3 neighborhood with submanifold mask), and
compares against the rdna4 kernel.
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from kernels import spconv_rdna4_ext as srx


def cpu_reference(feats, coords, weight, bias):
    """feats: [N, Ci], coords: [N, 4] (b,z,y,x), weight: [Co,3,3,3,Ci]"""
    N, Ci = feats.shape
    Co = weight.shape[0]
    # build coord -> idx dict
    table = {}
    for i in range(N):
        b, z, y, x = coords[i].tolist()
        table[(b, z, y, x)] = i
    out = torch.zeros(N, Co, dtype=feats.dtype, device=feats.device)
    if bias is not None:
        out += bias[None, :]
    # weight already in [Co, kd, kh, kw, Ci]; submanifold center at (1,1,1)
    for i in range(N):
        b, z, y, x = coords[i].tolist()
        for kd in range(3):
            for kh in range(3):
                for kw in range(3):
                    nz, ny, nx = z + (kd - 1), y + (kh - 1), x + (kw - 1)
                    j = table.get((b, nz, ny, nx))
                    if j is None:
                        continue
                    w = weight[:, kd, kh, kw, :]   # [Co, Ci]
                    out[i] += feats[j] @ w.T
    return out


def main():
    torch.manual_seed(0)
    device = 'cuda'
    # random sparse occupancy in a small grid
    G = 8
    coords = []
    for z in range(G):
        for y in range(G):
            for x in range(G):
                if torch.rand(1).item() < 0.4:
                    coords.append((0, z, y, x))
    coords_t = torch.tensor(coords, dtype=torch.int32, device=device)
    N = coords_t.shape[0]
    Ci, Co = 64, 128
    feats = torch.randn(N, Ci, device=device, dtype=torch.float32)
    weight = torch.randn(Co, 3, 3, 3, Ci, device=device, dtype=torch.float32) * 0.1
    bias = torch.randn(Co, device=device, dtype=torch.float32) * 0.1

    print(f'N={N}, Ci={Ci}, Co={Co}')

    # CPU reference (compute on cuda for matmul speed but indexed gather)
    ref = cpu_reference(feats, coords_t, weight, bias)

    # rdna4 ext
    out, nmap = srx.submanifold_conv3d(feats, coords_t, None, weight, bias=bias)

    abs_err = (out - ref).abs()
    rel = abs_err.max() / ref.abs().max().clamp_min(1e-6)
    print(f'max_abs_err={abs_err.max().item():.3e}  max_rel={rel.item():.3e}')
    print(f'ref range=[{ref.min().item():.3f}, {ref.max().item():.3f}]')
    print(f'out range=[{out.min().item():.3f}, {out.max().item():.3f}]')

    cos = torch.nn.functional.cosine_similarity(out.flatten().unsqueeze(0), ref.flatten().unsqueeze(0)).item()
    print(f'cos={cos:.6f}')

    # 2nd call uses cached nmap
    out2, nmap2 = srx.submanifold_conv3d(feats, coords_t, None, weight, bias=bias, neighbor_cache=nmap)
    print(f'cached_call_match={(out2 == out).all().item()}')

    # determinism: call 4 times, check identical
    outs = [srx.submanifold_conv3d(feats, coords_t, None, weight, bias=bias)[0] for _ in range(4)]
    diffs = [(o - outs[0]).abs().max().item() for o in outs]
    print(f'4-trial max diffs: {diffs}')

    assert cos > 0.9999, 'correctness FAILED'
    assert all(d == 0 for d in diffs), 'determinism FAILED'
    print('OK')


if __name__ == '__main__':
    main()
