#!/usr/bin/env python3
"""Pooled paired CUDA-vs-HIP closed-loop analysis.  usage: pool_pair.py task=cudaDir:hipDir[,hipDir2...] ...   (episodes paired by seed; a task may list several hip dirs, e.g. batches)
Prints per task and pooled: successes, discordant counts, exact McNemar p, paired success-rate difference (HIP - CUDA) with a Newcombe-Wilson 95% CI for paired proportions
and a seed-bootstrap CI, mean ticks-to-success, outcome agreement, and the N needed for an equivalence margin of +-10 points at the observed discordance rate."""
import glob, json, math, sys, random
def load(dirs):
    e = {}
    for d in dirs:
        for f in glob.glob(d + "/episode_*.json"):
            j = json.load(open(f)); e[j["seed"]] = j
    return e
def wilson(k, n, z=1.96):
    if n == 0: return 0.0, 1.0
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d; return c - h, c + h
def mcnemar(b, c):
    m = b + c; k = min(b, c)
    return 1.0 if m == 0 else min(1.0, 2 * sum(math.comb(m, i) for i in range(k + 1)) / 2 ** m)
def newcombe(a, b, c, d):  # paired: a both succ, b only-x, c only-y, d both fail ; diff = (c-b)/n for y-x
    n = a + b + c + d; p1 = (a + b) / n; p2 = (a + c) / n; l1, u1 = wilson(a + b, n); l2, u2 = wilson(a + c, n)
    den = math.sqrt((a + b) * (c + d) * (a + c) * (b + d)); phi = 0 if den == 0 else (a * d - b * c) / den
    phi = max(phi, 0.0)
    dl = (p2 - p1) - math.sqrt((p2 - l2) ** 2 + (u1 - p1) ** 2 - 2 * phi * (p2 - l2) * (u1 - p1))
    du = (p2 - p1) + math.sqrt((u2 - p2) ** 2 + (p1 - l1) ** 2 - 2 * phi * (u2 - p2) * (p1 - l1))
    return p2 - p1, dl, du
def stats(pairs):
    a = sum(x and y for x, y in pairs); b = sum(x and not y for x, y in pairs); c = sum(y and not x for x, y in pairs); d = sum((not x) and (not y) for x, y in pairs)
    n = len(pairs); diff, lo, hi = newcombe(a, b, c, d)
    rnd = random.Random(0); bs = []
    for _ in range(4000):
        s = [pairs[rnd.randrange(n)] for _ in range(n)]; bs.append(sum(y for x, y in s) / n - sum(x for x, y in s) / n)
    bs.sort(); disc = (b + c) / n
    # N for +-10pt equivalence: CI half-width ~ 1.96*sqrt(disc/N) (paired, small diff) <= 0.10
    need = math.ceil((1.96 ** 2) * disc / 0.10 ** 2) if disc else 0
    return dict(n=n, cuda=a + b, hip=a + c, only_cuda=b, only_hip=c, both=a, neither=d, p=mcnemar(b, c), diff=diff, nc=(lo, hi), boot=(bs[100], bs[3899]), agree=(a + d) / n, need=need)
if __name__ == "__main__":
    allp = []; tk = {"cuda": [], "hip": []}
    for spec in sys.argv[1:]:
        t, v = spec.split("="); cd, hd = v.split(":"); ec, eh = load([cd]), load(hd.split(","))
        seeds = sorted(set(ec) & set(eh)); pairs = [(ec[s]["success"], eh[s]["success"]) for s in seeds]; allp += pairs
        tk["cuda"] += [ec[s]["ticks"] for s in seeds if ec[s]["success"]]; tk["hip"] += [eh[s]["ticks"] for s in seeds if eh[s]["success"]]
        if not pairs: continue
        r = stats(pairs); print(f"{t:14s} n={r['n']:3d} CUDA {r['cuda']} HIP {r['hip']} | only-CUDA {r['only_cuda']} only-HIP {r['only_hip']} | McNemar p={r['p']:.3f} | diff(HIP-CUDA) {r['diff']*100:+.1f} pts Newcombe95 [{r['nc'][0]*100:+.1f},{r['nc'][1]*100:+.1f}] | agree {r['agree']*100:.0f}%")
    r = stats(allp)
    print(f"{'POOLED':14s} n={r['n']:3d} CUDA {r['cuda']} ({r['cuda']/r['n']*100:.0f}%) HIP {r['hip']} ({r['hip']/r['n']*100:.0f}%) | only-CUDA {r['only_cuda']} only-HIP {r['only_hip']} | McNemar p={r['p']:.3f} | diff {r['diff']*100:+.1f} pts Newcombe95 [{r['nc'][0]*100:+.1f},{r['nc'][1]*100:+.1f}] bootstrap95 [{r['boot'][0]*100:+.1f},{r['boot'][1]*100:+.1f}] | agree {r['agree']*100:.0f}% | N for +-10pt equivalence ~{r['need']}")
    print("mean ticks-to-success: CUDA %.0f (n=%d) HIP %.0f (n=%d)" % (sum(tk['cuda']) / max(1, len(tk['cuda'])), len(tk['cuda']), sum(tk['hip']) / max(1, len(tk['hip'])), len(tk['hip'])))
