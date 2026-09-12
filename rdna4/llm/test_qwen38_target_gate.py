"""Exercise the production footer parser without a model or GPU."""
from pathlib import Path
import os
import subprocess
import tempfile

root = Path(__file__).resolve().parent
(root / "tmp").mkdir(exist_ok=True)
with tempfile.TemporaryDirectory(prefix="qwen-target-gate-", dir=root / "tmp") as directory:
    log = Path(directory) / "gate.log"
    env = {k: v for k, v in os.environ.items() if not k.startswith(("QWEN38_", "LLM_"))}

    def check(rc, message, mode="", first="", hash_value=""):
        lines = []
        for i in range(3 if mode != "truncated" else 2):
            token = 31 if mode == "first" and i == 1 else 30
            value = "1123456789abcdef" if mode == "divergent" and i == 1 else "0123456789abcdef"
            lines.extend([f"First decoded token id={token}", f"sequence hash={value}",
                          "Prefill: 10 tokens -> 125.0 tok/s", "Decode: 4 tokens -> 20.0 tok/s",
                          "End-to-end: 14 tokens -> 100.0 tok/s", "Result: PASS"])
        log.write_text("\n".join(lines) + "\n")
        result = subprocess.run(["bash", "-euc",
                                 'source "$1"; qwen38_target_result "$2" 3 test "$3" "$4"',
                                 "test", str(root / "qwen38_target_result.sh"), str(log), first, hash_value],
                                env=env, text=True, capture_output=True, timeout=15)
        output = result.stdout + result.stderr
        assert result.returncode == rc, output
        assert message in output, output

    expected = dict(first="30", hash_value="0123456789abcdef")
    check(0, "target gate PASS")
    check(0, "reference parity: PASS", **expected)
    check(1, "scalar reference mismatch", **(expected | {"first": "31"}))
    check(1, "scalar reference mismatch", **(expected | {"hash_value": "1123456789abcdef"}))
    check(1, "nondeterministic sequence hash", mode="divergent", **expected)
    check(1, "nondeterministic first token", mode="first", **expected)
    check(1, "expected 3 sequence-hash footers", mode="truncated", **expected)
    fields = dict(QWEN38_TARGET_EXPECTED_FIRST_TOKEN="30", QWEN38_TARGET_EXPECTED_HASH="0123456789abcdef")
    for extra in ({"QWEN38_TARGET_EXPECTED_FIRST_TOKEN": "30"},
                  {"QWEN38_TARGET_EXPECTED_HASH": "0123456789abcdef"},
                  fields | {"QWEN38_TARGET_EXPECTED_HASH": "garbage"},
                  fields | {"QWEN38_TARGET_EXPECTED_FIRST_TOKEN": "-1"},
                  fields | {"QWEN38_TARGET_EXPECTED_FIRST_TOKEN": ""}):
        result = subprocess.run(["bash", str(root / "bench_qwen38_target.sh")],
                                env=env | {"QWEN38_DRY_RUN": "1"} | extra,
                                text=True, capture_output=True, timeout=15)
        assert result.returncode == 2 and "supply both" in result.stderr, result.stderr
print("Qwen38 target gate: repeatability, reference parity, malformed/truncated results: PASS")
