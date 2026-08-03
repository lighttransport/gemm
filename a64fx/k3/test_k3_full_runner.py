#!/usr/bin/env python3
import subprocess
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
RUNNER = HERE / "k3_full_runner"


class K3FullRunnerTest(unittest.TestCase):
    def _run(self, *args):
        return subprocess.run(
            [str(RUNNER), *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            check=False,
        )

    def test_help_lists_barrier_mode(self):
        proc = self._run("--help")
        self.assertEqual(proc.returncode, 0)
        self.assertIn("full96|layer12|synthetic12", proc.stderr)
        self.assertIn("--mode barrier", proc.stderr)
        self.assertIn("--barrier-iters", proc.stderr)
        self.assertIn("--comm-deterministic", proc.stderr)
        self.assertIn("--comm-bf16", proc.stderr)
        self.assertIn("--comm-poll-spins", proc.stderr)
        self.assertIn("--prefetch-mib", proc.stderr)
        self.assertIn("--profile", proc.stderr)
        self.assertIn("--ar-groups", proc.stderr)

    def test_barrier_iterations_are_positive(self):
        proc = self._run("--mode", "barrier", "--nodes", "12",
                         "--barrier-iters", "0")
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("expects [1,100000]", proc.stderr)

    def test_barrier_mode_does_not_require_model_paths(self):
        source = (HERE / "k3_full_runner.c").read_text()
        self.assertIn("if (o->mode == K3_FULL_MODE_BARRIER)", source)
        self.assertIn("full_barrier_stress(&opt)", source)

    def test_12_node_harness_runs_barrier_preflight(self):
        source = (HERE / "run_k3_full_12n.sh").read_text()
        self.assertIn('--mode barrier --nodes 12', source)
        self.assertIn('barrier.log', source)
        self.assertIn('COMM_DETERMINISTIC=${K3_COMM_DETERMINISTIC:-1}', source)
        self.assertIn('--comm-deterministic "$COMM_DETERMINISTIC"', source)
        self.assertIn('--expert-tp', source)
        self.assertIn('--q8', source)
        self.assertIn('full-convert', source)

    def test_batch_launchers_run_barrier_preflight(self):
        for name in ("pjsub_k3_full_96n_short_1h.sh", "pjsub_k3_full_96n.sh"):
            source = (HERE / name).read_text()
            self.assertIn('--mode barrier --nodes "$NODES"', source)
            self.assertIn('BARRIER_ITERS=${K3_BARRIER_ITERS:-128}', source)
            self.assertIn('#PJM -x K3_BARRIER_ITERS', source)
            self.assertIn('COMM_DETERMINISTIC=${K3_COMM_DETERMINISTIC:-1}', source)
            self.assertIn('--comm-deterministic "$COMM_DETERMINISTIC"', source)
            self.assertIn('#PJM -x K3_FULL_STAGE_DIR', source)
            self.assertIn('--ar-groups "$AR_GROUPS"', source)
            self.assertIn('K3_MOE_SHARD_LAYOUT', source)

    def test_full_runner_defaults_to_deterministic_reductions(self):
        source = (HERE / "k3_full_runner.c").read_text()
        self.assertIn('.comm_deterministic = 1', source)
        self.assertIn('.deterministic = opt.comm_deterministic', source)

    def test_full_runner_profiles_rank_maximum_layers_and_collectives(self):
        source = (HERE / "k3_full_runner.c").read_text()
        self.assertIn("full_profile_report", source)
        self.assertIn("rank_max_mean_ms", source)
        self.assertIn("full_profile_collective_add", source)

    def test_debug_layer_profile_reports_phases_and_latency_tail(self):
        source = (HERE / "k3_full_runner.c").read_text()
        debug = source.split("static int full_debug_layer_forward", 1)[1]
        debug = debug.split("static int full_synthetic_layer", 1)[0]
        for phase in ("K3_FULL_PHASE_ATTENTION", "K3_FULL_PHASE_MOE",
                      "K3_FULL_PHASE_RESIDUAL"):
            self.assertIn(phase, debug)
        self.assertIn("if (n == 1)", source)
        self.assertIn("layer_sum / active_layers", source)

    def test_full_runner_accepts_aligned_moe_shards(self):
        source = (HERE / "k3_full_runner.c").read_text()
        self.assertIn("full_split_groups(K3_LATENT", source)
        self.assertIn("full_split_groups(K3_HIDDEN", source)
        self.assertIn("invalid routed-up shard shape", source)

    def test_full_runner_has_q8_expert_tp_and_hierarchical_paths(self):
        source = (HERE / "k3_full_runner.c").read_text()
        self.assertIn("K3_FULL_DTYPE_Q8P16", source)
        self.assertIn("full_moe_forward_expert_tp", source)
        self.assertIn("tp_comm_init_2d_external", source)
        self.assertIn("tp_allreduce_sum_2d_checked", source)

    def test_q8_converter_uses_high_quality_router_layout(self):
        source = (HERE / "k3_full_convert.c").read_text()
        self.assertIn('block_sparse_moe.gate.weight', source)
        self.assertIn('k3_q8pv8_quantize_bf16', source)
        self.assertIn('router_q8 ? "Q8P8" : "Q8P16"', source)

    def test_q8_prepare_is_streaming_and_resumable(self):
        source = (HERE / "run_k3_full_q8_prepare_12n.sh").read_text()
        self.assertIn("logical+=12", source)
        self.assertIn("K3_Q8_REUSE", source)
        self.assertIn("--quality-gate", source)
        self.assertIn("native-full96-expert-tp", source)
        self.assertIn("MOE_SHARD_LAYOUT", source)
        self.assertIn("row-aligned", source)
        self.assertIn("export MODEL_DIR NATIVE_DIR MIXED_DIR QUALITY_GATE FORCE "
                      "SCRIPT_DIR MOE_SHARD_LAYOUT", source)

    def test_interactive_regression_matrix_is_model_free(self):
        source = (HERE / "run_k3_12n_regression.sh").read_text()
        self.assertIn("requires PJM_MPI_PROC=$NODES", source)
        self.assertIn("run_barrier_case startup 1", source)
        self.assertIn('run_barrier_case long "$LONG_BARRIER_ITERS"', source)
        for case in ("flat", "deterministic", "a2a", "bf16", "two-level", "ack-drop"):
            self.assertIn("run_ar_case " + case, source)
        self.assertIn("TP_AR_DROP", source)
        self.assertIn("--barrier-only", source)
        self.assertIn("--ar-only", source)


if __name__ == "__main__":
    unittest.main()
