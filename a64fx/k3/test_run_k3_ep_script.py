#!/usr/bin/env python3
import subprocess
import unittest
import tempfile
import os
from pathlib import Path

SCRIPT = str((Path(__file__).resolve().parent / "run_k3_ep.sh").resolve())


class RunK3EpScriptTest(unittest.TestCase):
    def _run(self, *args):
        env = os.environ.copy()
        env["PJM_MPI_PROC"] = "12"
        proc = subprocess.run(
            ["bash", SCRIPT, *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            env=env,
            timeout=5,
        )
        return proc.returncode, proc.stdout, proc.stderr

    def test_help_exits_zero(self):
        code, stdout, stderr = self._run("--help")
        self.assertEqual(code, 0)
        self.assertIn("usage:", stdout + stderr)

    def test_invalid_argument_rejected(self):
        code, stdout, stderr = self._run("--nodes", "12", "--bad", "x")
        self.assertNotEqual(code, 0)
        self.assertIn("unknown argument: --bad", (stderr + stdout))

    def test_mode_must_be_dummy_or_real(self):
        code, _stdout, stderr = self._run("--nodes", "12", "--tp-nodes", "6", "--mode", "weird")
        self.assertNotEqual(code, 0)
        self.assertIn("--mode must be dummy or real", stderr)

    def test_invalid_tp_nodes_rejected(self):
        code, _stdout, stderr = self._run("--nodes", "12", "--tp-nodes", "5")
        self.assertNotEqual(code, 0)
        self.assertIn("nodes in [1,512], tp-nodes in [1,96], and nodes divisible by tp-nodes", stderr)

    def test_stage_only_requires_real_mode(self):
        code, _stdout, stderr = self._run("--mode", "dummy", "--stage-only", "--nodes", "12", "--stage-dir", "/tmp")
        self.assertNotEqual(code, 0)
        self.assertIn("--stage-only requires --mode real", stderr)

    def test_numerics_reject_non_integer_tokens(self):
        code, _stdout, stderr = self._run("--nodes", "12", "--tp-nodes", "12", "--tokens", "abc")
        self.assertNotEqual(code, 0)
        self.assertIn("numeric options must be integers", stderr)

    def test_invalid_comm_robust(self):
        code, _stdout, stderr = self._run("--nodes", "12", "--tp-nodes", "12", "--comm-robust", "3")
        self.assertNotEqual(code, 0)
        self.assertIn("--comm-robust must be 1 or 2", stderr)

    def test_invalid_ar_groups(self):
        code, _stdout, stderr = self._run(
            "--nodes", "12",
            "--tp-nodes", "12",
            "--ar-groups", "5",
        )
        self.assertNotEqual(code, 0)
        self.assertIn("--ar-groups must be 0 or a divisor", stderr)

    def test_ar_groups_one_rejected_for_tp3(self):
        code, _stdout, stderr = self._run(
            "--nodes", "12",
            "--tp-nodes", "3",
            "--ar-groups", "1",
        )
        self.assertNotEqual(code, 0)
        self.assertIn("--ar-groups must be 0 or a divisor", stderr)

    def test_ar_groups_auto_is_accepted(self):
        with tempfile.TemporaryDirectory() as existing:
            code, _stdout, stderr = self._run(
                "--nodes", "12",
                "--tp-nodes", "4",
                "--ar-groups", "auto",
                "--result-dir", existing,
            )
        self.assertNotEqual(code, 0)
        self.assertIn("result directory already exists", stderr)

    def test_ar_groups_zero_is_flat(self):
        with tempfile.TemporaryDirectory() as existing:
            code, _stdout, stderr = self._run(
                "--nodes", "12",
                "--tp-nodes", "4",
                "--ar-groups", "0",
                "--result-dir", existing,
            )
        self.assertNotEqual(code, 0)
        self.assertIn("result directory already exists", stderr)

    def test_comm_poll_spins_must_be_power_of_two(self):
        code, _stdout, stderr = self._run(
            "--nodes", "12",
            "--tp-nodes", "12",
            "--comm-poll-spins", "3",
            "--result-dir", "/tmp/k3-comm-spins-bad",
        )
        self.assertNotEqual(code, 0)
        self.assertIn("--comm-poll-spins must be a power of two in [1,1024]", stderr)

    def test_invalid_prefetch_threads_limit(self):
        code, _stdout, stderr = self._run(
            "--nodes", "12",
            "--tp-nodes", "12",
            "--threads", "48",
            "--prefetch-threads", "49",
        )
        self.assertNotEqual(code, 0)
        self.assertIn("--prefetch-threads cannot exceed --threads", stderr)

    def test_prefetch_mib_requires_thread_cap(self):
        code, _stdout, stderr = self._run(
            "--nodes", "12",
            "--tp-nodes", "12",
            "--threads", "48",
            "--prefetch-mib", "1",
        )
        self.assertNotEqual(code, 0)
        self.assertIn("--prefetch-mib requires --threads <=47", stderr)

    def test_comm_ack_invalid_value_rejected(self):
        code, _stdout, stderr = self._run(
            "--nodes", "12",
            "--tp-nodes", "12",
            "--comm-ack", "2",
        )
        self.assertNotEqual(code, 0)
        self.assertIn("--comm-ack must be 0 or 1", stderr)

    def test_comm_deterministic_invalid_value_rejected(self):
        code, _stdout, stderr = self._run(
            "--nodes", "12",
            "--tp-nodes", "12",
            "--comm-deterministic", "2",
        )
        self.assertNotEqual(code, 0)
        self.assertIn("--comm-deterministic must be 0 or 1", stderr)

    def test_comm_deterministic_zero_is_accepted(self):
        with tempfile.TemporaryDirectory() as existing:
            code, _stdout, stderr = self._run(
                "--nodes", "12",
                "--tp-nodes", "12",
                "--comm-deterministic", "0",
                "--result-dir", existing,
            )
        self.assertNotEqual(code, 0)
        self.assertIn("result directory already exists", stderr)

    def test_comm_poll_spins_one_is_accepted(self):
        with tempfile.TemporaryDirectory() as existing:
            code, _stdout, stderr = self._run(
                "--nodes", "12",
                "--tp-nodes", "12",
                "--comm-poll-spins", "1",
                "--result-dir", existing,
            )
        self.assertNotEqual(code, 0)
        self.assertIn("result directory already exists", stderr)

    def test_comm_poll_spins_too_large_rejected(self):
        code, _stdout, stderr = self._run(
            "--nodes", "12",
            "--tp-nodes", "12",
            "--comm-poll-spins", "2048",
        )
        self.assertNotEqual(code, 0)
        self.assertIn("--comm-poll-spins must be a power of two in [1,1024]", stderr)

    def test_prefetch_mib_upper_bound(self):
        code, _stdout, stderr = self._run(
            "--nodes", "12",
            "--tp-nodes", "12",
            "--prefetch-mib", "33",
        )
        self.assertNotEqual(code, 0)
        self.assertIn("--prefetch-mib must be in [0,32]", stderr)

    def test_prefetch_mib_zero_is_accepted(self):
        with tempfile.TemporaryDirectory() as existing:
            code, _stdout, stderr = self._run(
                "--nodes", "12",
                "--tp-nodes", "12",
                "--prefetch-mib", "0",
                "--result-dir", existing,
            )
        self.assertNotEqual(code, 0)
        self.assertIn("result directory already exists", stderr)
        self.assertNotIn("prefetch-mib requires --threads <=47", stderr)

    def test_non_tmp_cache_paths_do_not_trigger_warning(self):
        with tempfile.TemporaryDirectory() as existing:
            code, _stdout, stderr = self._run(
                "--nodes", "12",
                "--tp-nodes", "12",
                "--cache-load", "/home/u14346/shared-load.bin",
                "--cache-save", "/var/tmp/shared-save.bin",
                "--result-dir", existing,
            )
        self.assertNotEqual(code, 0)
        self.assertIn("result directory already exists", stderr)
        self.assertNotIn("warning: --cache-load path is under /tmp", stderr)
        self.assertNotIn("warning: --cache-save path is under /tmp", stderr)

    def test_result_dir_exists(self):
        with tempfile.TemporaryDirectory() as existing:
            code, _stdout, stderr = self._run(
                "--nodes", "12",
                "--tp-nodes", "1",
                "--result-dir", existing,
            )
        self.assertNotEqual(code, 0)
        self.assertIn("result directory already exists", stderr)

    def test_cache_tmp_paths_warn_for_multinode_run(self):
        with tempfile.TemporaryDirectory() as existing:
            code, _stdout, stderr = self._run(
                "--nodes", "12",
                "--tp-nodes", "12",
                "--cache-load", "/tmp/k3-cache-load",
                "--cache-save", "/tmp/k3-cache-save",
                "--result-dir", existing,
            )
        self.assertNotEqual(code, 0)
        self.assertIn("warning: --cache-load path is under /tmp", stderr)
        self.assertIn("warning: --cache-save path is under /tmp", stderr)
        self.assertIn("result directory already exists", stderr)

    def test_prefetch_mib_thread_cap_forces_requirements(self):
        # threads=47 with prefetch_mib > 0 must fail fast in validation stage
        code, _stdout, stderr = self._run(
            "--nodes", "12",
            "--tp-nodes", "12",
            "--threads", "48",
            "--prefetch-mib", "1",
        )
        self.assertNotEqual(code, 0)
        self.assertIn("--prefetch-mib requires --threads <=47", stderr)

    def test_cache_load_and_save_paths_do_not_need_result_dir_collision(self):
        # With explicit unique result-dir, run script should fail later on existing dir check only
        with tempfile.TemporaryDirectory() as result:
            code, _stdout, stderr = self._run(
                "--nodes", "12",
                "--tp-nodes", "12",
                "--cache-load", "/home/u14346/does-not-matter.bin",
                "--cache-save", "/home/u14346/does-not-matter-save.bin",
                "--result-dir", result,
            )
        self.assertNotEqual(code, 0)
        self.assertIn("result directory already exists", stderr)

    def test_cache_paths_on_tmp_are_warning_predecessor(self):
        with tempfile.TemporaryDirectory() as result:
            code, _stdout, stderr = self._run(
                "--nodes", "12",
                "--tp-nodes", "12",
                "--cache-load", "/tmp/k3-cache-load",
                "--cache-save", "/tmp/k3-cache-save",
                "--result-dir", result,
            )
        self.assertNotEqual(code, 0)
        self.assertIn("warning: --cache-load path is under /tmp", stderr)
        self.assertIn("warning: --cache-save path is under /tmp", stderr)

    def test_cache_paths_on_local_warn_for_multinode_run(self):
        with tempfile.TemporaryDirectory() as result:
            code, _stdout, stderr = self._run(
                "--nodes", "12",
                "--tp-nodes", "12",
                "--cache-load", "/local/k3-cache-load",
                "--cache-save", "/local/k3-cache-save",
                "--result-dir", result,
            )
        self.assertNotEqual(code, 0)
        self.assertIn("warning: --cache-load path is under /local", stderr)
        self.assertIn("warning: --cache-save path is under /local", stderr)
        self.assertIn("result directory already exists", stderr)


if __name__ == "__main__":
    unittest.main()
