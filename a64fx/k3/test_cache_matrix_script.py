#!/usr/bin/env python3
import os
import stat
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("run_12n_cache_matrix.sh")


class CacheMatrixScriptTest(unittest.TestCase):
    def _fake_runner(self, directory, missing_rank=False, bad_result=False):
        runner = Path(directory) / "fake_runner.sh"
        runner.write_text(textwrap.dedent(f"""\
            #!/bin/sh
            set -eu
            result=
            nodes=12
            tokens=0
            cache_load=0
            cache_save=0
            cache_save_path=
            while [ "$#" -gt 0 ]; do
                case "$1" in
                    --result-dir) result=$2; shift 2 ;;
                    --nodes) nodes=$2; shift 2 ;;
                    --tokens) tokens=$2; shift 2 ;;
                    --cache-load) cache_load=1; shift 2 ;;
                    --cache-save) cache_save=1; cache_save_path=$2; shift 2 ;;
                    *) shift ;;
                esac
            done
            completed=$tokens
            if [ "$cache_load" -eq 1 ] && [ "$cache_save" -eq 0 ]; then
                completed=0
            fi
            disagreement={"1.000e+00" if bad_result else "0.000e+00"}
            mkdir -p "$result"
            if [ "$cache_save" -eq 1 ]; then
                mkdir -p "$cache_save_path"
            fi
            i=0
            passes=0
            while [ "$i" -lt "$nodes" ]; do
                if [ "{1 if missing_rank else 0}" -eq 0 ] || [ "$i" -ne 0 ]; then
                    printf 'rank=%s state=pass\\n' "$i" > "$result/k3_rank$(printf '%03d' "$i").status"
                    printf 'K3_RESULT status=PASS reason=complete tokens_completed=%s disagreement=%s\\n' "$completed" "$disagreement" > "$result/rank.fake.$i"
                    if [ "$cache_save" -eq 1 ]; then
                        : > "$cache_save_path/k3_ep_cache_fake_$i.bin"
                    fi
                    passes=$((passes + 1))
                fi
                i=$((i + 1))
            done
            printf 'K3 distributed result: rc=0 pass_markers=%s/%s results=%s\\n' "$passes" "$nodes" "$result"
        """))
        runner.chmod(runner.stat().st_mode | stat.S_IXUSR)
        return runner

    def _run(self, runner, result):
        env = os.environ.copy()
        env.update({
            "NODES": "12", "TP_NODES": "12", "THREADS": "8",
            "RESULT_ROOT": str(result), "RUNNER": str(runner),
            "CACHE_TOKENS": "32", "TOKENS": "64",
        })
        return subprocess.run(
            ["bash", str(SCRIPT)], stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, universal_newlines=True, env=env, timeout=10,
        )

    def test_lifecycle_enforces_rank_results_and_zero_token_restore(self):
        with tempfile.TemporaryDirectory() as directory:
            result = Path(directory) / "result"
            proc = self._run(self._fake_runner(directory), result)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("K3_MATRIX_END status=PASS", proc.stdout)

    def test_missing_rank_result_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            result = Path(directory) / "result"
            proc = self._run(self._fake_runner(directory, missing_rank=True), result)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("reason=rank-markers", proc.stderr)

    def test_nonzero_disagreement_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            result = Path(directory) / "result"
            proc = self._run(self._fake_runner(directory, bad_result=True), result)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("reason=result-invariant", proc.stderr)


if __name__ == "__main__":
    unittest.main()
