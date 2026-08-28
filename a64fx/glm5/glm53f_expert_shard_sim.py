#!/usr/bin/env python3
"""Monte-Carlo slowest-rank load for GLM-5.3F intra-expert sharding."""
import argparse
import itertools
import random


def simulate(parts, trials, seed):
    rng = random.Random(seed)
    maxima = []
    for _ in range(trials):
        load = [0] * 12
        for expert in rng.sample(range(288), 8):
            base = expert % 12
            for part in range(parts):
                load[(base + part * (12 // parts)) % 12] += 1
        maxima.append(max(load))
    maxima.sort()
    return sum(maxima) / trials, maxima[trials // 2], maxima[int(0.95 * trials)], maxima[int(0.99 * trials)]


def simulate_offsets(offsets, samples):
    maxima = []
    for experts in samples:
        load = [0] * 12
        for expert in experts:
            for offset in offsets:
                load[(expert + offset) % 12] += 1
        maxima.append(max(load))
    maxima.sort()
    n = len(maxima)
    return sum(maxima) / n, maxima[n // 2], maxima[int(.95 * n)], maxima[int(.99 * n)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=53)
    parser.add_argument("--search-offsets", action="store_true")
    args = parser.parse_args()
    # A64FX job 51040571: fused gate_up + down, 47 threads.
    measured_ms = {1: 0.2109, 2: 0.1219, 4: 0.0724}
    for parts in (1, 2, 4):
        mean, p50, p95, p99 = simulate(parts, args.trials, args.seed)
        print("parts=%d mean_max_units=%.4f p50=%d p95=%d p99=%d mean_critical_ms=%.4f" %
              (parts, mean, p50, p95, p99, mean * measured_ms[parts]))
    if args.search_offsets:
        rng = random.Random(args.seed)
        samples = [rng.sample(range(288), 8) for _ in range(args.trials)]
        ranked = []
        # Translation does not affect load, so fix the first owner offset at zero.
        for tail in itertools.combinations(range(1, 12), 3):
            offsets = (0,) + tail
            ranked.append((simulate_offsets(offsets, samples), offsets))
        for stats, offsets in sorted(ranked)[:10]:
            print("offsets=%s mean_max_units=%.4f p50=%d p95=%d p99=%d" %
                  (offsets, stats[0], stats[1], stats[2], stats[3]))


if __name__ == "__main__":
    main()
