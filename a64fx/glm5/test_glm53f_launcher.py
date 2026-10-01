"""Launcher contract tests; no model, compiler, MPI allocation, or /tmp needed."""
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

MODULE = Path(__file__).resolve().parent
REPO = MODULE.parent.parent

MOCK_MPI = r'''#!/usr/bin/env python3
import os, subprocess, sys
args = sys.argv[1:]
assert args[:2] == ['-n', '12'], args
args = args[2:]
prefix = None
if args[:1] == ['-of-proc']:
    prefix, args = args[1], args[2:]
for rank in range(12):
    env = dict(os.environ, PMIX_RANK=str(rank))
    if prefix:
        with open(prefix + '.mock.' + str(rank), 'w') as output:
            rc = subprocess.call(args, env=env, stdout=output, stderr=output)
    else:
        rc = subprocess.call(args, env=env)
    if rc: sys.exit(rc)
'''

MOCK_PROGRAM = r'''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
name = Path(sys.argv[0]).name
rank = int(os.environ.get('PMIX_RANK', '0'))
with open(os.environ['MOCK_CALLS'], 'a') as log:
    log.write(json.dumps({'name': name, 'args': sys.argv[1:], 'rank': rank,
        'native': os.environ.get('GLM53F_Q2_KDA_STAGE'),
        'core': os.environ.get('GLM53F_REPACK_DIR')}) + '\n')
if name == 'tofu_topo_helper':
    Path('tofu_topo.txt').write_text('\n'.join(str(i) for i in range(12)) + '\n')
elif name == 'glm53f_core_stage':
    print('SENTINEL glm53f_rank_stage=OK')
elif name == 'glm53f_core_add_routers':
    print('GLM53F_CORE_ROUTERS PASS')
elif name == 'glm53f_decode_stage':
    status = Path(os.environ['GLM53F_STATUS_DIR']) / ('rank%02d.status' % rank)
    status.write_text('fixture OK\n')
    print('STAGE PASS')
elif name.startswith('glm53f_q2_'):
    if str(rank) != os.environ.get('MOCK_MISSING_SENTINEL'):
        print('SENTINEL ' + name + '=OK')
elif name == 'glm53f_target_decode_12n':
    mode = 'GENERATE' if sys.argv[4] == '--generate' else 'DECODE'
    print('GLM53F_TARGET_' + mode + '_12N steps=2 tok_s=30.000 PASS')
    if mode == 'GENERATE': print('GLM53F_TARGET_TIMING prompt_tok_s=40 decode_tok_s=30')
elif name == 'glm53f_spec_decode_12n':
    print('GLM53F_SPEC_REFERENCE PASS')
    if not os.environ.get('MOCK_SPEC_INCOMPLETE'):
        print('GLM53F_SPEC_COMPLETE {"status":"PASS"}')
elif name == 'test_glm53f_lookup_spec':
    print('GLM53F_LOOKUP_SPEC PASS cases=28')
elif name == 'glm53f_executor_check_12n':
    print('GLM53F_EXECUTOR_CHECK tokens=32 state=BIT_EXACT PASS')
elif name == 'bench_glm53f_run_12n':
    if not os.environ.get('MOCK_BENCH_INCOMPLETE'):
        print('GLM53F_BENCH_COMPLETE {"status":"PASS"}')
elif name == 'test_glm53f_state_io':
    if len(sys.argv) != 2 or not Path(sys.argv[1]).is_dir(): sys.exit(2)
    print('STATE_IO PASS')
else:
    print(name + ' PASS')
'''


class LauncherTest(unittest.TestCase):
    def setUp(self):
        (REPO / 'tmp').mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix='glm53f-launcher-test-', dir=str(REPO / 'tmp'))
        self.root = Path(self.temp.name)
        self.bin = self.root / 'bin'
        self.bin.mkdir()
        self.calls = self.root / 'calls.jsonl'
        self.env = {k: v for k, v in os.environ.items()
                    if not k.startswith(('GLM53F_', 'PJM_', 'TOFU_', 'PMIX_'))}
        self.env.update(PJM_JOBID='test', PJM_NODE='12', PJM_MPI_PROC='12',
                        GLM53F_BUILD='0', GLM53F_BIN_DIR=str(self.bin),
                        GLM53F_BUILD_DIR=str(self.root / 'objects'),
                        GLM53F_LOG_DIR=str(self.root / 'logs'), MOCK_CALLS=str(self.calls))
        mpi = self.root / 'mpiexec'
        mpi.write_text(MOCK_MPI)
        mpi.chmod(0o755)
        self.env['GLM53F_MPIEXEC'] = str(mpi)
        program = self.bin / 'mock-program'
        program.write_text(MOCK_PROGRAM)
        program.chmod(0o755)
        names = ['glm53f_target_decode_12n', 'tofu_topo_helper', 'glm53f_core_stage',
                 'glm53f_core_add_routers', 'glm53f_decode_stage']
        names += ['test_glm53f_' + kind for kind in ('kquant', 'native_batch', 'prefill_config', 'state_io',
                                                    'team', 'mhc_team', 'iq_grouped', 'lookup', 'lookup_spec', 'moe_combine', 'index_heads')]
        names += ['bench_glm53f_run_12n']
        names += ['glm53f_' + kind for kind in ('kda_callback_check', 'dense_batch_check', 'sparse_batch_check',
                                               'target_batch_check_12n', 'executor_check_12n', 'spec_decode_12n')]
        names += ['glm53f_q2_' + kind for kind in
                  ('stage', 'embed_stage', 'head_stage', 'dense_stage', 'sparse_stage',
                   'kda_stage', 'shexp_stage', 'core_patch', 'shared_patch')]
        for name in names:
            (self.bin / name).symlink_to(program)
        for key in ('STAGE_DIR', 'GGUF_SHARED_STAGE', 'GGUF_CORE_STAGE', 'SHARED_STAGE_DIR',
                    'REPACK_STAGE_DIR', 'Q2_EMBED_STAGE', 'Q2_HEAD_STAGE', 'Q2_DENSE_STAGE',
                    'Q2_SPARSE_STAGE', 'Q2_KDA_STAGE', 'Q2_SHEXP_STAGE', 'MTP_STAGE_DIR', 'MTP_SHARED_STAGE_DIR'):
            path = self.root / key
            path.mkdir()
            suffix = '.core' if 'CORE' in key or 'REPACK' in key else ''
            for rank in range(12):
                (path / ('rank%02d%s.manifest' % (rank, suffix))).write_text('fixture\n')
            self.env['GLM53F_' + key] = str(path)
        self.prompt = self.root / 'prompt with spaces.ids'
        self.prompt.write_text('1 42\n')

    def tearDown(self):
        self.temp.cleanup()

    def run_cli(self, *args, **overrides):
        env = dict(self.env, **overrides)
        return subprocess.run(['bash', str(MODULE / 'run_glm53f_12n.sh')] + list(args),
                              cwd=str(REPO), env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, universal_newlines=True, timeout=30)

    def records(self):
        return [json.loads(line) for line in self.calls.read_text().splitlines()] if self.calls.exists() else []

    def test_help_without_allocation(self):
        result = self.run_cli('--help', PJM_MPI_PROC='0')
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertFalse(self.records())

    def test_documentation_links(self):
        for name in ('README.md', 'GLM53F_VALIDATION.md', 'GLM53F_EXPERIMENTS.md'):
            for link in re.findall(r'\]\(([^)]+)\)', (MODULE / name).read_text()):
                if '://' not in link and not link.startswith('#'):
                    self.assertTrue((MODULE / link.split('#')[0]).exists(), (name, link))

    def test_decode_reuses_stages_and_forwards_options(self):
        result = self.run_cli('decode', '42', '2', '--decode-window', '1')
        self.assertEqual(result.returncode, 0, result.stdout)
        calls = [c for c in self.records() if c['name'] == 'glm53f_target_decode_12n']
        self.assertEqual(len(calls), 12)
        self.assertEqual(calls[0]['args'][3:], ['42', '2', '--decode-window', '1'])
        self.assertEqual(calls[0]['native'], self.env['GLM53F_Q2_KDA_STAGE'])
        self.assertFalse(any('_stage' in c['name'] for c in self.records()))

    def test_executor_check_forwards_prompt_and_bounded_steps(self):
        result = self.run_cli('executor-check', str(self.prompt), '32')
        self.assertEqual(result.returncode, 0, result.stdout)
        calls = [c for c in self.records() if c['name'] == 'glm53f_executor_check_12n']
        self.assertEqual(len(calls), 12)
        self.assertEqual(calls[0]['args'][3], str(self.prompt))
        self.assertEqual(calls[0]['args'][5], '32')
        self.assertNotEqual(self.run_cli('executor-check', str(self.prompt), '513').returncode, 0)

    def test_generate_paths_and_option_order(self):
        output = str(self.root / 'output with spaces.ids')
        result = self.run_cli('generate', str(self.prompt), output, '2', '--prefill-chunk', '32')
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn('GLM53F_TARGET_TIMING', result.stdout)
        call = next(c for c in self.records() if c['name'] == 'glm53f_target_decode_12n')
        self.assertEqual(call['args'][3:], ['--generate', str(self.prompt), output, '2', '--prefill-chunk', '32'])

    def test_invalid_inputs_fail_before_mpi(self):
        for args in [('generate', str(self.prompt), str(self.prompt), '2'),
                     ('generate', str(self.prompt), 'out.ids', '0'), ('decode', '1', '0'),
                     ('decode', '154880', '2'), ('unknown',)]:
            self.assertNotEqual(self.run_cli(*args).returncode, 0, args)
        self.assertNotEqual(self.run_cli('decode', PJM_NODE='1').returncode, 0)
        self.assertNotEqual(self.run_cli('decode', GLM53F_BIN_DIR='/local/missing-bin').returncode, 0)
        self.assertNotEqual(self.run_cli('decode', GLM53F_RUN_TAG='../escape').returncode, 0)
        self.assertFalse(self.records())

    def test_benchmark_records_inputs_and_forwards_runtime_options(self):
        output = str(self.root / 'benchmark with spaces.ids')
        result = self.run_cli('benchmark', str(self.prompt), output, '--transitions', '256',
                              '--decode-executor', 'persistent', SECRET_DO_NOT_LOG='private-value')
        self.assertEqual(result.returncode, 0, result.stdout)
        calls = [c for c in self.records() if c['name'] == 'bench_glm53f_run_12n']
        self.assertEqual(len(calls), 12)
        self.assertEqual(calls[0]['args'][3:], [str(self.prompt), output, '--transitions', '256',
                                              '--decode-executor', 'persistent'])
        files = list(Path(self.env['GLM53F_LOG_DIR']).glob('benchmark-metadata-*'))
        self.assertEqual(len(files), 1)
        metadata = files[0].read_text()
        self.assertIn('OMP_NUM_THREADS=47', metadata)
        self.assertNotIn('private-value', metadata)

    def test_benchmark_rejects_existing_output_and_incomplete_run(self):
        output = self.root / 'existing.ids'
        output.write_text('preserve\n')
        self.assertNotEqual(self.run_cli('benchmark', str(self.prompt), str(output)).returncode, 0)
        self.assertEqual(output.read_text(), 'preserve\n')
        self.assertFalse(self.records())
        result = self.run_cli('benchmark', str(self.prompt), str(self.root / 'new.ids'), MOCK_BENCH_INCOMPLETE='1')
        self.assertNotEqual(result.returncode, 0)

    def test_missing_rank_manifest_prevents_loading(self):
        (Path(self.env['GLM53F_Q2_KDA_STAGE']) / 'rank07.manifest').unlink()
        result = self.run_cli('decode')
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.records())

    def test_hybrid_clears_inherited_native_stages(self):
        result = self.run_cli('decode', '1', '2', GLM53F_NATIVE='0')
        self.assertEqual(result.returncode, 0, result.stdout)
        call = next(c for c in self.records() if c['name'] == 'glm53f_target_decode_12n')
        self.assertIsNone(call['native'])
        self.assertEqual(call['core'], self.env['GLM53F_REPACK_STAGE_DIR'])

    def test_check_runs_component_and_full_model_gates(self):
        result = self.run_cli('check')
        self.assertEqual(result.returncode, 0, result.stdout)
        calls = self.records()
        state = next(c for c in calls if c['name'] == 'test_glm53f_state_io')
        self.assertEqual(state['args'], [self.env['GLM53F_LOG_DIR']])
        self.assertEqual(sum(c['name'] == 'glm53f_dense_batch_check' for c in calls), 36)
        target = next(c for c in calls if c['name'] == 'glm53f_target_batch_check_12n')
        self.assertIn('--prefill-mode', target['args'])

    def test_mtp_reuses_native_target_contract(self):
        result = subprocess.run(['bash', str(MODULE / 'run_glm53f_q4_mtp_12n.sh'),
                                 str(self.prompt), str(self.root / 'mtp.ids'), '1', '1'],
                                cwd=str(REPO), env=self.env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, universal_newlines=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout)
        call = next(c for c in self.records() if c['name'] == 'glm53f_spec_decode_12n')
        self.assertEqual(call['native'], self.env['GLM53F_Q2_KDA_STAGE'])
        self.assertEqual(call['args'][2], self.env['GLM53F_GGUF_SHARED_STAGE'])

    def test_mtp_forwards_resident_trial_options(self):
        result = subprocess.run(['bash', str(MODULE / 'run_glm53f_q4_mtp_12n.sh'),
                                 str(self.prompt), str(self.root / 'mtp.ids'), '128', '4',
                                 '--repetitions', '3', '--draft-sweep', '--ignore-eos', '--verify-kernel', 'grouped'],
                                cwd=str(REPO), env=self.env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, universal_newlines=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout)
        call = next(c for c in self.records() if c['name'] == 'glm53f_spec_decode_12n')
        self.assertEqual(call['args'][5:], ['1', '128', '4', '0', '--repetitions', '3',
                                           '--draft-sweep', '--ignore-eos', '--verify-kernel', 'grouped'])

    def test_mtp_rejects_missing_completion(self):
        result = subprocess.run(['bash', str(MODULE / 'run_glm53f_q4_mtp_12n.sh'),
                                 str(self.prompt), str(self.root / 'mtp.ids'), '1', '1'],
                                cwd=str(REPO), env=dict(self.env, MOCK_SPEC_INCOMPLETE='1'),
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=30)
        self.assertNotEqual(result.returncode, 0, result.stdout)

    def test_mtp_stage_reads_checkpoint_without_target_repack(self):
        result = subprocess.run(['bash', str(MODULE / 'run_glm53f_mtp_stage_12n.sh')],
                                cwd=str(REPO), env=self.env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, universal_newlines=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout)
        calls = self.records()
        self.assertEqual(len(calls), 24)
        self.assertTrue(all(c['core'] is None for c in calls))

    def test_native_stage_uses_bounded_copier_and_twelve_sentinels(self):
        gguf = self.root / 'model.gguf'
        gguf.write_text('fixture')
        overrides = dict(GLM53F_GGUF=str(gguf), GLM53F_GGUF_CORE_STAGE='/local/glm53f-mock-core',
                         GLM53F_GGUF_SHARED_STAGE='/local/glm53f-mock-shared')
        unsafe = dict(overrides, GLM53F_GGUF_CORE_STAGE='/local/../')
        self.assertNotEqual(self.run_cli('stage', **unsafe).returncode, 0)
        self.assertFalse(self.records())
        result = self.run_cli('stage', **overrides)
        self.assertEqual(result.returncode, 0, result.stdout)
        copies = [c for c in self.records() if c['name'] == 'glm53f_core_stage']
        self.assertEqual(len(copies), 24)
        self.assertEqual({c['args'][-1] for c in copies}, {'core', 'model'})
        result = self.run_cli('stage', MOCK_MISSING_SENTINEL='11', GLM53F_RUN_TAG='missing', **overrides)
        self.assertNotEqual(result.returncode, 0)


if __name__ == '__main__':
    unittest.main()
