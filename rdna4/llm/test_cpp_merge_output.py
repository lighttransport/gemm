#!/usr/bin/env python3
"""Compile and execute merge_intervals implementations from generation logs."""

import argparse
import hashlib
import os
from pathlib import Path
import re
import subprocess
import tempfile


PRELUDE = r'''#include <algorithm>
#include <cassert>
#include <climits>
#include <cstddef>
#include <iostream>
#include <random>
#include <utility>
#include <vector>
'''

HARNESS = r'''
using Interval = std::pair<int, int>;

static std::vector<Interval> reference_merge(std::vector<Interval> input) {
    std::sort(input.begin(), input.end());
    std::vector<Interval> output;
    for (const auto &item : input) {
        if (output.empty() || item.first > output.back().second) {
            output.push_back(item);
        } else if (item.second > output.back().second) {
            output.back().second = item.second;
        }
    }
    return output;
}

static void check(const std::vector<Interval> &input,
                  const std::vector<Interval> &expected) {
    const auto got = merge_intervals(input);
    if (got != expected) {
        std::cerr << "mismatch\n";
        std::abort();
    }
}

int main() {
    check({}, {});
    check({{4, 9}}, {{4, 9}});
    check({{3, 4}, {1, 2}}, {{1, 2}, {3, 4}});
    check({{5, 7}, {1, 3}, {2, 4}, {10, 10}, {10, 12}},
          {{1, 4}, {5, 7}, {10, 12}});
    check({{1, 10}, {2, 3}, {2, 3}, {4, 8}, {10, 10}}, {{1, 10}});
    check({{INT_MAX, INT_MAX}, {INT_MIN, -1}, {-1, 0},
           {1, INT_MAX - 1}},
          {{INT_MIN, 0}, {1, INT_MAX - 1}, {INT_MAX, INT_MAX}});

    std::mt19937 rng(0x7319u);
    std::uniform_int_distribution<int> count_dist(0, 40);
    std::uniform_int_distribution<int> value_dist(-1000, 1000);
    for (int trial = 0; trial < 10000; ++trial) {
        std::vector<Interval> input;
        const int count = count_dist(rng);
        for (int i = 0; i < count; ++i) {
            int a = value_dist(rng), b = value_dist(rng);
            if (a > b) std::swap(a, b);
            input.emplace_back(a, b);
        }
        if ((trial % 97) == 0) {
            input.emplace_back(INT_MIN, INT_MIN);
            input.emplace_back(INT_MAX, INT_MAX);
        }
        check(input, reference_merge(input));
    }
    std::cout << "PASS: fixed edge cases + 10000 randomized cases\n";
}
'''


def generated_text(path):
    text = path.read_text(encoding="utf-8", errors="strict")
    match = re.search(r"=== Generated text ===\n(.*?)\n=== end ===", text,
                      re.S)
    if not match:
        raise RuntimeError(f"{path}: generated response missing")
    body = match.group(1)
    markers = ("Ġ", "Ċ", "�", "<|im_start|>", "<|im_end|>",
               "<|endoftext|>", "```")
    leaked = [marker for marker in markers if marker in body]
    if leaked:
        raise RuntimeError(f"{path}: leaked or disallowed markers {leaked}")
    return body


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--require-identical", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    tmp_root = Path(os.environ.get("TMPDIR", repo_root / "tmp"))
    tmp_root.mkdir(parents=True, exist_ok=True)
    bodies = []
    run_env = dict(os.environ)
    # LeakSanitizer cannot inspect a process under some agent sandboxes.
    run_env["ASAN_OPTIONS"] = "detect_leaks=0"

    with tempfile.TemporaryDirectory(prefix="cpp-merge-output-",
                                     dir=tmp_root) as tmp:
        work = Path(tmp)
        for index, log in enumerate(args.logs):
            body = generated_text(log)
            bodies.append(body)
            source = work / f"case-{index}.cpp"
            binary = work / f"case-{index}"
            source.write_text(PRELUDE + body + "\n" + HARNESS,
                              encoding="utf-8")
            subprocess.run([
                "g++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                "-pedantic", "-D_GLIBCXX_ASSERTIONS",
                "-fsanitize=address,undefined", "-fno-omit-frame-pointer",
                str(source), "-o", str(binary),
            ], check=True)
            result = subprocess.check_output(
                [str(binary)], text=True, env=run_env).strip()
            digest = hashlib.sha256(body.encode()).hexdigest()
            print(f"{log}: {result}; sha256={digest}")

    if args.require_identical and len(set(bodies)) != 1:
        raise RuntimeError(f"responses differ: {len(set(bodies))} unique bodies")
    if args.require_identical:
        print(f"MATCH: all {len(bodies)} generated responses are byte-identical")


if __name__ == "__main__":
    main()
