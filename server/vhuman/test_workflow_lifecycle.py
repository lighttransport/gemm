"""Regression coverage for cancellation, rig publication and concurrent uploads."""
from concurrent.futures import ThreadPoolExecutor
import io
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

from PIL import Image

from . import gpu, process
from .rig.job import rig_job
from .service import EyeService, ROOT


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        scratch = ROOT/'tmp/test-workflow-lifecycle'
        scratch.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.service = EyeService(self.root)

    @unittest.skipUnless(shutil.which('node'), 'Node.js unavailable')
    def test_rig_workspace_request_races_and_candidate_actions(self):
        result = subprocess.run(['node', str(Path(__file__).with_name('test_rig_workspace.cjs'))],
                                cwd=ROOT, capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)

    def test_cancel_kills_child_after_parent_exit_and_releases_stdout(self):
        self._worker_cleanup(cancel=True)

    def test_progress_exception_also_kills_worker_group(self):
        self._worker_cleanup(cancel=False)

    def test_body_cancellation_cleans_up_descendants(self):
        self._worker_cleanup(cancel=True, runner='body')

    def test_speech_cancellation_cleans_up_descendants(self):
        self._worker_cleanup(cancel=True, runner='speech')

    def test_timeout_kills_descendant_holding_stdout_after_parent_exit(self):
        child = "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)"
        command = [sys.executable, '-c',
                   'import subprocess,sys; subprocess.Popen([sys.executable,"-c",sys.argv[1]])', child]
        with self.assertRaises(subprocess.TimeoutExpired):
            process.run(command, threading.Event(), lambda _: None, cwd=ROOT,
                        env=dict(os.environ), timeout=.4)

    def _worker_cleanup(self, *, cancel, runner=None):
        child = self.root/'child.py'
        child.write_text("import signal,time\nsignal.signal(signal.SIGTERM,signal.SIG_IGN)\n"
                         "print('ready',flush=True)\ntime.sleep(60)\n")
        code = ("import subprocess,sys,signal,time; signal.signal(signal.SIGTERM,signal.SIG_DFL); "
                "subprocess.Popen([sys.executable,sys.argv[1]]); time.sleep(60)")
        event = threading.Event()
        errors, children = [], []
        popen = subprocess.Popen

        def launch(*args, **kwargs):
            proc = popen(*args, **kwargs)
            children.append(proc)
            return proc

        def on_line(line):
            if line == 'ready':
                if cancel:
                    event.set()
                else:
                    raise ValueError('progress failed')

        def work():
            try:
                cmd = [sys.executable, '-u', '-c', code, str(child)]
                if runner:
                    from .body.job import _run as body_run
                    from .rig.speech import _run as speech_run
                    threading.Timer(.4, event.set).start()
                    if runner == 'body':
                        body_run(cmd, event)
                    else:
                        speech_run(cmd, event, lambda *_: None)
                else:
                    process.run(cmd, event, on_line, cwd=ROOT, env=dict(os.environ))
            except Exception as exc:
                errors.append(exc)

        with patch.object(process.subprocess, 'Popen', side_effect=launch):
            thread = threading.Thread(target=work, daemon=True)
            thread.start()
            try:
                thread.join(8)
                self.assertFalse(thread.is_alive(), 'worker blocked on inherited stdout')
                self.assertEqual(len(errors), 1)
                self.assertIsInstance(errors[0], gpu.Cancelled if cancel else ValueError)
                self.assertEqual(children[0].poll(), -signal.SIGTERM)
                # Check that the pipe-draining thread has also finished.
                self.assertTrue(children[0].stdout.closed)
            finally:
                for proc in children:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                thread.join(5)

    def _body_rebuild(self, fail_publication=False, cancel_assembly=False):
        from .body import job
        rig = self._head(False)
        (rig/'features.json').write_text('{}')
        body = rig.parent/'body'
        take = body/'motions/012345abcdef'
        take.mkdir(parents=True)
        (body/'avatar.json').write_text('old avatar')
        (body/'avatar.glb').write_bytes(b'old glb')
        manifest = {'id': take.name, 'avatar_sha256': hashlib.sha256(b'old avatar').hexdigest()[:16]}
        (take/'manifest.json').write_text(json.dumps(manifest))
        (take/'motion.json').write_bytes(b'keep motion')
        self.assertFalse(self.service.list_motions('abc')[0]['avatar_stale'])

        def assemble(cmd, *args, **kwargs):
            out = Path(cmd[cmd.index('--out')+1])
            (out/'body_report.json').write_text('{}')
            (out/'avatar.json').write_text('new avatar')
            (out/'avatar.glb').write_bytes(b'new glb')
            if cancel_assembly:
                args[0].set()

        rename = Path.rename
        def publish(path, target):
            if fail_publication and path.name.startswith('.body-') and target == body:
                # Allow restoration of the saved package.
                if not path.name.startswith('.body-previous-'):
                    raise OSError('publication failed')
            return rename(path, target)

        with gpu.execution('cpu'), patch.object(job.gpu, 'gpu_status', return_value=None), \
                patch.object(job, 'availability', return_value={'available': True}), \
                patch.object(job, '_generate_image', return_value={'bbox': [0, 0, 1, 1]}), \
                patch.object(job, '_sam_body', return_value={}), \
                patch.object(job, '_pixal', return_value={}), \
                patch.object(job, '_garments', return_value=[]), \
                patch.object(job, '_run', side_effect=assemble), patch.object(Path, 'rename', publish):
            call = lambda: job.body_job(self.service, {'head_id': 'abc'}, lambda *_: None,
                                       threading.Event(), mock=True)
            if cancel_assembly:
                with self.assertRaises(gpu.Cancelled):
                    call()
            elif fail_publication:
                with self.assertRaisesRegex(OSError, 'publication failed'):
                    call()
            else:
                call()
        self.assertEqual((take/'motion.json').read_bytes(), b'keep motion')
        kept_old = fail_publication or cancel_assembly
        self.assertEqual(self.service.list_motions('abc')[0]['avatar_stale'], not kept_old)
        self.assertEqual((body/'avatar.json').read_text(), 'old avatar' if kept_old else 'new avatar')
        self.assertFalse(list(body.parent.glob('.body-*')))

    def test_body_rebuild_preserves_motion_downloads_and_marks_stale(self):
        self._body_rebuild()
        self.assertEqual(self.service.body_file('abc', 'motions/012345abcdef/motion.json').read_bytes(), b'keep motion')

    def test_body_publication_failure_restores_motions(self):
        self._body_rebuild(fail_publication=True)

    def test_body_cancelled_after_assembly_preserves_avatar_and_motions(self):
        self._body_rebuild(cancel_assembly=True)

    def test_cpu_reconstruction_job_never_probes_or_locks_gpu(self):
        from .reconstruction.job import reconstruction_job
        self._head(False)
        commands=[]
        def worker(command,*args,**kwargs):
            commands.append(command)
            folder=Path(command[command.index('--module')+2])
            run=command[command.index('--run-id')+1]
            destination=folder/'reconstruction'/run;destination.mkdir(parents=True)
            (destination/'manifest.json').write_text('{}')
            return 0
        with gpu.execution('cpu'),patch.object(gpu,'gpu_status',side_effect=AssertionError('GPU discovery')), \
                patch.object(gpu,'device_session',side_effect=AssertionError('GPU session')), \
                patch.object(process,'run',side_effect=worker):
            result=reconstruction_job(self.service,dict(head_id='abc',threads=2,iterations=20),
                                      lambda *_:None,threading.Event(),python=sys.executable)
        self.assertTrue(result['run_id'])
        self.assertEqual(commands[0][commands[0].index('--backend')+1],'cpu')
        self.assertEqual(commands[0][commands[0].index('--threads')+1],'2')

    def test_video_take_detects_changed_native_package_with_same_controls(self):
        rig=self._head(False)
        (rig/'rig_deformer.safetensors').write_bytes(b'old native package')
        take=rig/'takes/012345abcdef';take.mkdir()
        manifest=dict(id=take.name,rig_sha256=hashlib.sha256((rig/'rig.json').read_bytes()).hexdigest()[:16],
                      rig_package_sha256=hashlib.sha256(b'old native package').hexdigest())
        (take/'manifest.json').write_text(json.dumps(manifest))
        (take/'motion.npz').write_bytes(b'motion')
        self.assertFalse(self.service.take_summary('abc',take.name)['rig_stale'])
        self.assertIn('gnm_motion',self.service.take_summary('abc',take.name)['urls'])
        (rig/'rig_deformer.safetensors').write_bytes(b'new native package')
        self.assertTrue(self.service.take_summary('abc',take.name)['rig_stale'])

    def test_motion_staleness_checks_geometry_and_body_metadata(self):
        self._body_rebuild()
        body = self.root/'heads/abc/body'
        manifest = body/'motions/012345abcdef/manifest.json'
        record = json.loads(manifest.read_text())
        record['avatar_sha256'] = hashlib.sha256((body/'avatar.json').read_bytes()).hexdigest()
        record['avatar_glb_sha256'] = hashlib.sha256((body/'avatar.glb').read_bytes()).hexdigest()
        (body/'body_mhr.glb.json').write_text('{}')
        record['body_meta_sha256'] = hashlib.sha256(b'{}').hexdigest()
        manifest.write_text(json.dumps(record))
        self.assertFalse(self.service.list_motions('abc')[0]['avatar_stale'])
        for filename in ('avatar.glb', 'body_mhr.glb.json'):
            with self.subTest(filename=filename):
                path = body/filename; old = path.read_bytes(); path.write_bytes(b'changed')
                self.assertTrue(self.service.list_motions('abc')[0]['avatar_stale'])
                path.write_bytes(old)

    def _head(self, candidate):
        head = self.root/'heads'/'abc'
        head.mkdir(parents=True, exist_ok=True)
        for name in ('head.json', 'fit.json', 'head_eyes.glb', 'portrait.png'):
            (head/name).write_text('{}')
        if candidate:
            head = head/'reconstruction'/'def'
            head.mkdir(parents=True, exist_ok=True)
            (head/'manifest.json').write_text('{"face_model":"gnm_v3"}')
        rig = head/'rig'
        (rig/'takes'/'take').mkdir(parents=True)
        (rig/'takes'/'take'/'audio.wav').write_bytes(b'keep user take')
        (rig/'rig.json').write_text('old controls')
        (rig/'rig.glb').write_text('old mesh')
        (rig/'rig_report.json').write_text('{"face_model":"gnm_v3"}')
        return rig

    def _rebuild(self, candidate, runner):
        request = dict(head_id='abc', res=1024, iters=50)
        if candidate:
            request['reconstruction_run'] = 'def'
        with gpu.execution('cpu'), patch.object(process, 'run', side_effect=runner):
            return rig_job(self.service, request, lambda *_: None, threading.Event(),
                           python=sys.executable, mock=True)

    def test_failed_accepted_and_candidate_rebuild_preserve_complete_old_rig(self):
        for candidate in (False, True):
            with self.subTest(candidate=candidate):
                rig = self._head(candidate)
                before = {p.relative_to(rig): p.read_bytes() for p in rig.rglob('*') if p.is_file()}

                def fail(cmd, *args, **kwargs):
                    out = Path(cmd[cmd.index('--out')+1])
                    out.mkdir()
                    (out/'rig.json').write_text('new controls')
                    return 1

                with self.assertRaisesRegex(RuntimeError, 'rig build failed'):
                    self._rebuild(candidate, fail)
                self.assertEqual(before, {p.relative_to(rig): p.read_bytes() for p in rig.rglob('*') if p.is_file()})
                self.assertFalse(list(rig.parent.glob('.rig-*')))

    def test_successful_rebuild_preserves_takes_and_updates_entire_rig(self):
        rig = self._head(False)

        def success(cmd, *args, **kwargs):
            out = Path(cmd[cmd.index('--out')+1])
            out.mkdir()
            for name in ('rig.json', 'rig.glb'):
                (out/name).write_text('new '+name)
            (out/'rig_report.json').write_text('{"seconds":1}')
            # The previous package remains readable throughout export.
            self.assertEqual((rig/'rig.json').read_text(), 'old controls')
            return 0

        result = self._rebuild(False, success)
        self.assertEqual((rig/'rig.glb').read_text(), 'new rig.glb')
        self.assertEqual((rig/'rig.json').read_text(), 'new rig.json')
        self.assertEqual((rig/'takes/take/audio.wav').read_bytes(), b'keep user take')
        self.assertEqual(result['glb_url'], '/v1/heads/abc/rig/rig.glb')
        self.assertFalse(list(rig.parent.glob('.rig-*')))

    def test_cancelled_accepted_rebuild_preserves_old_package(self):
        rig = self._head(False)

        def cancel_build(cmd, event, *args, **kwargs):
            out = Path(cmd[cmd.index('--out')+1]); out.mkdir()
            (out/'rig_report.json').write_text('{}')
            event.set()
            return 0

        with self.assertRaises(gpu.Cancelled):
            self._rebuild(False, cancel_build)
        self.assertEqual((rig/'rig.json').read_text(), 'old controls')
        self.assertEqual((rig/'takes/take/audio.wav').read_bytes(), b'keep user take')
        self.assertFalse(list(rig.parent.glob('.rig-*')))

    def test_publication_failure_restores_old_rig_and_takes(self):
        rig = self._head(False)
        replace = Path.replace

        def fail_publish(path, target):
            if path.name.endswith('.partial') and target == rig:
                raise OSError('publication failed')
            return replace(path, target)

        def success(cmd, *args, **kwargs):
            out = Path(cmd[cmd.index('--out')+1]); out.mkdir()
            (out/'rig_report.json').write_text('{}')
            return 0

        with patch.object(Path, 'replace', fail_publish):
            with self.assertRaisesRegex(OSError, 'publication failed'):
                self._rebuild(False, success)
        self.assertEqual((rig/'rig.json').read_text(), 'old controls')
        self.assertEqual((rig/'takes/take/audio.wav').read_bytes(), b'keep user take')
        self.assertFalse(list(rig.parent.glob('.rig-*')))

    def test_concurrent_identical_portrait_uploads_both_succeed(self):
        from .reconstruction.upload import upload
        image = io.BytesIO(); Image.new('RGB', (64, 64), 'red').save(image, format='PNG')
        data = image.getvalue()
        entered, written = threading.Barrier(2), threading.Barrier(2)
        write = Path.write_bytes

        def overlap(path, value):
            entered.wait(5)
            result = write(path, value)
            written.wait(5)
            return result

        with patch.object(Path, 'write_bytes', overlap), ThreadPoolExecutor(2) as pool:
            futures = [pool.submit(upload, self.service, io.BytesIO(data), len(data), 'image/png') for _ in range(2)]
            results = [f.result(timeout=10) for f in futures]
        self.assertEqual(results[0], results[1])
        path = self.root/'portrait_uploads'/(results[0]['upload_id']+'.png')
        with Image.open(path) as im:
            im.load()
            self.assertEqual(im.size, (64, 64))
        self.assertFalse(list(path.parent.glob('*.partial')))


if __name__ == '__main__':
    unittest.main()
