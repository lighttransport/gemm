"""Summarise run dirs: success + Wilson 95% CI, mean ticks to success, per-call latency, boundary jump / step CV (jitter.py logic), peak VRAM.
usage: summarize.py <run dir>...   |   summarize.py --pair <dirA> <dirB>  (paired by seed: exact McNemar on discordant pairs)"""
import glob, json, math, os, statistics as st, sys
import numpy as np

def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"),) * 2
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h

def load(d):
    return {json.load(open(f))["seed"]: json.load(open(f)) for f in sorted(glob.glob(d + "/episode_*.json"))}

def jitter(d):
    cv, bj, med = [], [], []
    for f in sorted(glob.glob(d + "/episode_*.npz")):
        z = np.load(f); cp = z["cmd_pose"][:, :3] * 1e3; c = z["call"]
        for k in range(c.max() + 1):
            i = np.where(c == k)[0]; s = np.linalg.norm(np.diff(cp[i], axis=0), axis=1)
            if len(s) > 2: cv.append(s.std() / s.mean())
            med += list(s)
            if k > 0: bj.append(np.linalg.norm(cp[i[0]] - cp[i[0] - 1]))
    return np.mean(cv), np.mean(bj), np.median(med)

def row(d):
    e = load(d); n = len(e); k = sum(v["success"] for v in e.values()); lo, hi = wilson(k, n)
    ts = [v["ticks"] for v in e.values() if v["success"]]
    calls = [x for v in e.values() for x in v["call_s"]]
    try:
        cj = json.load(open(d + "/calls.json")); pk = max(c["peak_gb_sample"] for c in cj["calls"]); pe = max((c["peak_gb_encode"] or 0) for c in cj["calls"])
    except Exception: pk = pe = float("nan")
    cv, bj, md = jitter(d)
    return dict(name=os.path.basename(d), k=k, n=n, lo=lo, hi=hi, ticks=st.mean(ts) if ts else float("nan"), call=st.mean(calls), p50=st.median(calls),
                bj=bj, cv=cv, peak=pk, peake=pe, fails=sorted(s for s, v in e.items() if not v["success"]))

def mcnemar(a, b):
    ea, eb = load(a), load(b); seeds = sorted(set(ea) & set(eb))
    b01 = sum(ea[s]["success"] and not eb[s]["success"] for s in seeds); b10 = sum(eb[s]["success"] and not ea[s]["success"] for s in seeds)
    m = b01 + b10; kk = min(b01, b10)
    p = 1.0 if m == 0 else min(1.0, 2 * sum(math.comb(m, i) for i in range(kk + 1)) / 2 ** m)
    return len(seeds), b01, b10, p

if __name__ == "__main__":
    if sys.argv[1] == "--pair":
        n, x, y, p = mcnemar(sys.argv[2], sys.argv[3]); print(f"paired n={n}: only-first-succeeds {x}, only-second-succeeds {y}, exact McNemar p={p:.3f}")
    else:
        print("| run | success | Wilson 95% CI | mean ticks (succ) | call mean/p50 s | boundary jump mm | step CV | peak GB sample/encode | failed seeds |\n|---|---|---|---|---|---|---|---|---|")
        for d in sys.argv[1:]:
            r = row(d)
            print(f"| {r['name']} | {r['k']}/{r['n']} | {r['lo']*100:.0f}-{r['hi']*100:.0f}% | {r['ticks']:.0f} | {r['call']:.2f}/{r['p50']:.2f} | {r['bj']:.1f} | {r['cv']:.2f} | {r['peak']:.2f}/{r['peake']:.2f} | {r['fails']} |")
