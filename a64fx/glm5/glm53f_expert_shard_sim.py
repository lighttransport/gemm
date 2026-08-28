#!/usr/bin/env python3
"""Monte-Carlo slowest-rank load for GLM-5.3F intra-expert sharding."""
import argparse
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=53)
    args = parser.parse_args()
    # A64FX job 51040571: fused gate_up + down, 47 threads.
    measured_ms = {1: 0.2109, 2: 0.1219, 4: 0.0724}
    for parts in (1, 2, 4):
        mean, p50, p95, p99 = simulate(parts, args.trials, args.seed)
        print("parts=%d mean_max_units=%.4f p50=%d p95=%d p99=%d mean_critical_ms=%.4f" %
              (parts, mean, p50, p95, p99, mean * measured_ms[parts]))


if __name__ == "__main__":
    main()
