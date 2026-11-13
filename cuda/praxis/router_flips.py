"""Router sensitivity: per MoE layer, fraction of tokens whose top-k expert SET is identical to the reference dump, mean overlap |A&B|/k,
and rel-L2 of the router logits (router input is the fp32 residual stream; the router matmul itself is never quantised).
  ~/.venv/bin/python -I cuda/praxis/router_flips.py --ref tmp/praxis_ref/real_fp32 --ours tmp/praxis_attr/bf16 [--ours2 ...]"""
import argparse, numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("--ref", required=True); ap.add_argument("--ours", nargs="+", required=True)
a = ap.parse_args()
layers = [("enc", i) for i in range(2, 14)] + [("dec", i) for i in range(1, 6)]
for d in a.ours:
    print(f"== {d} vs {a.ref}: same-set% / mean overlap / router-logit rel-L2")
    ss = []
    for t, i in layers:
        r = np.load(f"{a.ref}/{t}_b{i}_top_idx.npy").astype(int); o = np.load(f"{d}/{t}_b{i}_top_idx.npy").astype(int)
        n = min(len(r), len(o)); r, o = r[:n], o[:n]
        same = np.mean([set(x) == set(y) for x, y in zip(r, o)])
        ov = np.mean([len(set(x) & set(y)) / len(x) for x, y in zip(r, o)])
        lr = np.load(f"{a.ref}/{t}_b{i}_router_logits.npy")[:n].astype(np.float64); lo = np.load(f"{d}/{t}_b{i}_router_logits.npy")[:n].astype(np.float64)
        rl = np.linalg.norm(lr - lo) / np.linalg.norm(lr)
        ss.append((same, ov, rl))
        print(f"  {t}{i:<2d} same-set {100*same:5.1f}%  overlap {ov:.3f}  logits rel-L2 {rl:.2e}")
    s = np.array(ss); print(f"  MEAN same-set {100*s[:,0].mean():.1f}%  overlap {s[:,1].mean():.3f}  logits {s[:,2].mean():.2e}")
