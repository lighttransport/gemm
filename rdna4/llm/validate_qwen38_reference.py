#!/usr/bin/env python3
"""Compare complete Q8/Q8 responses, then time fresh uncached resident repeats.

Build the pinned oracle with build_llama_reference.sh first. Each process owns
the GPU exclusively. Trace I/O is excluded from the separate timing runs.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from compare_generation import read_trace


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def cpp_prompt():
    """Exactly 4096 tokens with the pinned Qwen3.8-27B tokenizer."""
    return (
        "<|im_start|>user\n"
        "Background reference material follows. The final Task section is the only instruction to answer.\n"
        + "Reliable library code handles empty inputs, boundary integer values, deterministic ordering, and avoids overflow-prone arithmetic.\n" * 171
        + "Reliable library code handles\n\n"
        "Task: The caller has already included <vector>, <utility>, <algorithm>, <cstddef>, and <climits>. Write only the concise C++17 function definition below, with no Markdown, explanation, or main function:\n"
        "std::vector<std::pair<int, int>> merge_intervals(std::vector<std::pair<int, int>> intervals);\n"
        "Each input pair satisfies first <= second. Sort by first, then merge intervals that overlap inclusively (next.first <= current.second). Do not merge intervals that are merely adjacent. Handle empty input, duplicates, nested intervals, INT_MIN, and INT_MAX without arithmetic overflow.\n"
        "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, help="default: built-in 4096-token C++ merge task")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=Path("tmp/qwen38/reference-build"))
    parser.add_argument("--runner", type=Path, help="runner binary (default: rdna4/llm/test_hip_llm)")
    parser.add_argument("--scratch", type=Path, help="shared HIPRTC cache/scratch (default: tmp/qwen38/hiprtc-cache)")
    parser.add_argument("--reuse-reference", type=Path,
                        help="reuse a prior run's matching pinned reference traces and timing logs")
    parser.add_argument("--decode", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--native-q8-attn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--native-q8-prefill", action="store_true")
    parser.add_argument("--native-q2k", action="store_true")
    parser.add_argument("--native-mmvq", action="store_true")
    parser.add_argument("--mtp", type=Path, help="dense NextN sidecar, with exact target window verification")
    parser.add_argument("--mtp-draft", type=int, default=3)
    parser.add_argument("--dflash2", type=Path,
                        help="DFlash2 sidecar; greedy target windows and sampled target fallback")
    parser.add_argument("--dflash2-draft", type=int, default=7)
    parser.add_argument("--cpp-merge", action="store_true")
    args = parser.parse_args()
    if args.decode < 1 or args.decode > 4096 or args.repeats < 2:
        parser.error("decode must be 1..4096 and repeats at least 2 (cold + warm)")
    root = Path(__file__).resolve().parents[2]
    args.model = args.model.resolve()
    if args.mtp:
        args.mtp = args.mtp.resolve()
        if not 1 <= args.mtp_draft <= 15:
            parser.error("MTP draft width must be 1..15")
    if args.dflash2:
        args.dflash2 = args.dflash2.resolve()
        if not 1 <= args.dflash2_draft <= 7:
            parser.error("DFlash2 draft width must be 1..7")
    if args.mtp and args.dflash2:
        parser.error("MTP and DFlash2 sidecars are mutually exclusive")
    args.reference, args.out = args.reference.resolve(), args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    if args.prompt is None:
        args.prompt = args.out / "prompt-4096.txt"
        args.prompt.write_text(cpp_prompt())
        args.cpp_merge = True
    args.prompt = args.prompt.resolve()
    scratch = args.scratch.resolve() if args.scratch else root / "tmp/qwen38/hiprtc-cache"
    scratch.mkdir(parents=True, exist_ok=True)
    runner = args.runner.resolve() if args.runner else root / "rdna4/llm/test_hip_llm"
    oracle = args.reference / "llama_reference"
    reference_manifest = json.loads((args.reference / "manifest.json").read_text())
    for file, expected in reference_manifest["sha256"].items():
        if digest(Path(file)) != expected:
            raise RuntimeError(f"reference artifact changed: {file}")
    env = dict(os.environ, TMPDIR=str(scratch), QWEN38_RUNNER_BIN=str(runner),
               QWEN38_MODEL=str(args.model), LLM_GEN_TEXT="1")
    manifest = {"reference": reference_manifest, "sha256": {}, "commands": [],
                "shape": {"prompt": 4096, "chunk": 512, "context": 8192, "kv": "q8q8"},
                "environment": {k: v for k, v in env.items() if k == "TMPDIR" or k.startswith(("QWEN38_", "LLM_", "MM_BLASLT_", "HIP_RUNNER_"))}}
    for path in (runner, args.model, args.prompt):
        manifest["sha256"][str(path)] = digest(path)
    if args.mtp:
        manifest["sha256"][str(args.mtp)] = digest(args.mtp)
    if args.dflash2:
        manifest["sha256"][str(args.dflash2)] = digest(args.dflash2)
    previous = None
    if args.reuse_reference:
        args.reuse_reference = args.reuse_reference.resolve()
        previous_path = args.reuse_reference / "manifest.json"
        previous = json.loads(previous_path.read_text())
        previous_result_path = args.reuse_reference / "result.json"
        previous_result = json.loads(previous_result_path.read_text())
        if any("llama" not in previous_result.get(mode, {}).get("performance", {})
               for mode in ("greedy", "sampled")):
            raise RuntimeError("reused reference run did not complete both modes")
        if previous["reference"] != reference_manifest:
            raise RuntimeError("reused reference build manifest differs")
        if previous["sha256"].get(str(args.model)) != manifest["sha256"][str(args.model)]:
            raise RuntimeError("reused reference model differs")
        manifest["reused_reference"] = {"manifest": str(previous_path), "sha256": digest(previous_path),
                                        "result": str(previous_result_path), "result_sha256": digest(previous_result_path)}
    result = {}
    failed = False

    def run(command, log):
        manifest["commands"].append({"argv": [str(x) for x in command], "log": str(log)})
        (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print("RUN", log.name, flush=True)
        with log.open("wb") as f:
            proc = subprocess.run(command, cwd=root, env=env, stdout=f, stderr=subprocess.STDOUT)
        if proc.returncode:
            raise RuntimeError(f"{log}: process exited {proc.returncode}")
        if "Result: PASS" not in log.read_text(errors="replace"):
            raise RuntimeError(f"{log}: success marker missing")

    for mode in ("greedy", "sampled"):
        settings = ["--temp", "0" if mode == "greedy" else "0.6", "--seed", "42",
                    "--top-k", "20", "--top-p", "0.95", "--min-p", "0",
                    "--repeat-penalty", "1", "--frequency-penalty", "0", "--presence-penalty", "0"]
        commands = {
            "ours": ["bash", str(root / "rdna4/llm/run_qwen38_gsq_rocm.sh"), "--gpu-only-bench",
                     "--prompt-file", str(args.prompt), "-n", "4096", "-s", "8192", "--ubatch", "512",
                     "--kv-cache", "q8q8", "--qwen35-prefill-bf16", "--qwen35-decode-graph",
                     "--decode", str(args.decode), "--sampling-profile", "llama"],
            "llama": [str(oracle), str(args.model), str(args.prompt), "--decode", str(args.decode),
                      "--ctx", "8192", "--ubatch", "512"],
        }
        if args.native_q8_attn:
            commands["ours"].append("--qwen35-native-q8-attn")
        if args.native_q8_prefill:
            commands["ours"].append("--qwen35-native-q8-prefill")
        if args.native_q2k:
            commands["ours"].append("--qwen35-native-q2k")
        if args.native_mmvq:
            commands["ours"].append("--qwen35-native-mmvq")
        if args.mtp:
            commands["ours"] += ["--qwen35-mtp", str(args.mtp), "--qwen35-mtp-draft",
                                 str(args.mtp_draft), "--qwen35-mtp-window"]
        if args.dflash2:
            commands["ours"] += ["--qwen35-dflash2", str(args.dflash2),
                                 "--qwen35-dflash2-draft", str(args.dflash2_draft)]
        prefixes = {}
        for backend, command in commands.items():
            reuse = backend == "llama" and previous is not None
            prefix = (args.reuse_reference if reuse else args.out) / f"{backend}-{mode}"
            prefixes[backend] = prefix
            if reuse:
                saved = next((entry["argv"] for entry in previous["commands"]
                              if Path(entry["log"]).name == f"llama-{mode}.log"), None)
                if not saved or previous["sha256"].get(saved[2]) != manifest["sha256"][str(args.prompt)]:
                    raise RuntimeError("reused reference prompt differs")
                saved = list(saved)
                saved[2] = str(args.prompt)
                expected = [str(x) for x in command + settings + ["--trace-prefix", str(prefix)]]
                if saved != expected:
                    raise RuntimeError("reused reference generation settings differ")
                for suffix in (".log", ".prompt.tokens", ".0.tokens", ".0.output", ".0.logits.f32"):
                    path = Path(f"{prefix}{suffix}")
                    manifest["sha256"][str(path)] = digest(path)
                print("REUSE", prefix, flush=True)
            else:
                run(command + settings + ["--trace-prefix", str(prefix)], Path(f"{prefix}.log"))
        comparison = subprocess.run([sys.executable, str(root / "rdna4/llm/compare_generation.py"),
            str(prefixes["ours"]), str(prefixes["llama"])], capture_output=True, text=True)
        if comparison.returncode not in (0, 1) or not comparison.stdout.strip():
            raise RuntimeError(comparison.stderr)
        report = json.loads(comparison.stdout)
        failed |= comparison.returncode != 0
        result[mode] = {"comparison": report, "performance": {}}
        if args.cpp_merge:
            subprocess.run([sys.executable, str(root / "rdna4/llm/test_cpp_merge_output.py"),
                str(prefixes["ours"]) + ".log", str(prefixes["llama"]) + ".log"], cwd=root, env=env, check=True)
        for backend, command in commands.items():
            reuse = backend == "llama" and previous is not None
            log = (args.reuse_reference if reuse else args.out) / f"{backend}-{mode}-performance.log"
            repeat_flag = "--bench-repeat" if backend == "ours" else "--repeat"
            if reuse:
                expected = [str(x) for x in command + settings + [repeat_flag, str(args.repeats)]]
                saved = next((entry["argv"] for entry in previous["commands"]
                              if Path(entry["log"]).name == log.name), None)
                if not saved or previous["sha256"].get(saved[2]) != manifest["sha256"][str(args.prompt)]:
                    raise RuntimeError("reused reference timing prompt differs")
                saved = list(saved)
                saved[2] = str(args.prompt)
                if saved != expected:
                    raise RuntimeError("reused reference timing settings differ")
                manifest["sha256"][str(log)] = digest(log)
            else:
                run(command + settings + [repeat_flag, str(args.repeats)], log)
            text = log.read_text(encoding="utf-8", errors="strict")
            if "Result: PASS" not in text or "Result: FAIL" in text:
                raise RuntimeError(f"{log}: timing generation failed")
            outputs = re.findall(r"=== Generated text ===\n(.*?)\n=== end ===", text, re.S)
            prompt, _, expected, finish, _ = read_trace(prefixes[backend], 0)
            if len(prompt) != 4096:
                raise RuntimeError(f"{backend}: expected exactly 4096 prompt tokens")
            if len(outputs) != args.repeats or any(x.encode() != expected for x in outputs):
                raise RuntimeError(f"{log}: repeat output differs from traced response")
            finishes = re.findall(r"GENERATION finish=(\w+) selected=(\d+) emitted=(\d+) synthetic=(\d+)", text)
            if len(finishes) != args.repeats or any(x[0] != finish or x[3] != "0" for x in finishes):
                raise RuntimeError(f"{log}: inconsistent finish accounting")
            if backend == "ours":
                prefill = [float(x) for x in re.findall(r"Prefill:.*?-> ([\d.]+) tok/s", text)]
                decode = [float(x) for x in re.findall(r"Decode:.*?-> ([\d.]+) tok/s", text)]
            else:
                prefill = [float(x) for x in re.findall(r"PREFILL .*?tok_s=([\d.]+)", text)]
                decode = [float(x) for x in re.findall(r"DECODE .*?tok_s=([\d.]+)", text)]
            if len(prefill) != args.repeats or len(decode) != args.repeats:
                raise RuntimeError(f"{log}: timing records missing")
            result[mode]["performance"][backend] = {"prefill_tok_s": prefill, "decode_tok_s": decode,
                "reused": reuse,
                "warm_prefill_target_met": min(prefill[1:]) >= 500,
                "warm_decode_target_met": min(decode[1:]) >= 40}
        (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
