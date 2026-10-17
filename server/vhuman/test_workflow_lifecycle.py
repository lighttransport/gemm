"""Regression coverage for cancellation, rig publication and concurrent uploads."""
from concurrent.futures import ThreadPoolExecutor
import io
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

    def _worker_cleanup(self, *, cancel):
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
                process.run([sys.executable, '-u', '-c', code, str(child)], event,
                            on_line, cwd=ROOT, env=dict(os.environ))
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
