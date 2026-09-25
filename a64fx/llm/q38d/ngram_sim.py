import re, sys
ids = [int(m.group(1)) for m in re.finditer(r'token n=\d+ pos=\d+ id=(\d+)', open(sys.argv[1]).read())]
def sim(ids, k, nmax=4, nmin=1):
    # prompt-lookup: at step i (history ids[:i]), find the latest earlier occurrence of the longest
    # suffix (n = nmax..nmin) and propose the k tokens that followed it. Verify: accept prefix matches
    # of the true continuation, plus the one token the verify pass produces anyway.
    i, passes, drafted_ok = 1, 0, 0
    while i < len(ids):
        prop = []
        for n in range(nmax, nmin - 1, -1):
            if i < n: continue
            suf = ids[i - n:i]
            for j in range(i - n - 1, -1, -1):
                if ids[j:j + n] == suf:
                    prop = ids[j + n:j + n + k]
                    break
            if prop: break
        acc = 0
        while acc < len(prop) and i + acc < len(ids) and prop[acc] == ids[i + acc]: acc += 1
        passes += 1; drafted_ok += acc
        i += acc + 1
    return (len(ids) - 1) / passes, drafted_ok
for k in (1, 2, 3, 4, 6, 8):
    tpp, ok = sim(ids, k)
    print(f"k={k}: tokens/verify-pass={tpp:.2f} accepted drafts={ok}")
