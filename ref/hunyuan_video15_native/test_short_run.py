"""Deadline cleanup must finish before the suspended baseline is resumed."""
from contextlib import contextmanager,nullcontext
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import unittest
from unittest.mock import patch
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('hv15n_short_run',HERE/'short_run.py')
runner=importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
SCRATCH=runner.ROOT/'tmp/hv15-native/tests'
SCRATCH.mkdir(parents=True,exist_ok=True)


class Sampler:
    vram=rss=0
    def start(self,pid): self.pid=pid
    def close(self): pass


class ShortRun(unittest.TestCase):
    def invoke(self,command,seconds=2,check_descendant=False):
        with tempfile.TemporaryDirectory(dir=SCRATCH) as directory:
            report=Path(directory)/'receipt.json'
            released=[]
            @contextmanager
            def reservation(*args):
                try: yield
                finally:
                    # Every Popen must have been waited before resuming a baseline.
                    for process in spawned: self.assertIsNotNone(process.poll())
                    released.append(True)
                    if check_descendant:
                        pid=int(report.with_suffix('.log').read_text().strip())
                        try:
                            state=(Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()[0]
                        except FileNotFoundError:
                            state='X'
                        if state not in ('Z','X'):
                            os.kill(pid,signal.SIGKILL)
                        self.assertIn(state,('Z','X'),'descendant still running when reservation released')
            spawned=[]
            original=runner.subprocess.Popen
            def spawn(*args,**kwargs):
                process=original(*args,**kwargs);spawned.append(process);return process
            argv=['short_run','--out',str(report),'--seconds',str(seconds),'--',sys.executable,'-c',command]
            with patch.object(sys,'argv',argv),patch.object(runner,'experiment_lock',lambda cancel:nullcontext()),patch.object(runner,'gpu_reservation',reservation),patch.object(runner,'MemorySampler',Sampler),patch.object(runner.subprocess,'Popen',spawn):
                code=runner.main()
            result=json.loads(report.read_text())
            self.assertTrue(released)
            self.assertLess(result['elapsed_seconds'],seconds)
            return code,result

    def test_timeout_kills_before_reservation_exit(self):
        code,result=self.invoke('import time;time.sleep(10)',1)
        self.assertEqual(code,1)
        self.assertEqual(result['status'],'timeout')

    def test_crash_releases_and_fails(self):
        code,result=self.invoke('raise RuntimeError("probe failed")')
        self.assertEqual(code,1)
        self.assertEqual(result['status'],'failed')

    def test_success(self):
        code,result=self.invoke('print("PASS")')
        self.assertEqual(code,0)
        self.assertEqual(result['status'],'pass')

    def test_exited_leader_cannot_leave_descendant(self):
        command='import subprocess,sys;child=subprocess.Popen([sys.executable,"-c","import time;time.sleep(30)"]);print(child.pid,flush=True)'
        code,result=self.invoke(command,check_descendant=True)
        self.assertEqual(code,0)
        self.assertEqual(result['status'],'pass')

    def test_contended_experiment_lock_honors_deadline(self):
        import threading
        with tempfile.TemporaryDirectory(dir=SCRATCH) as directory,patch.object(runner,'ROOT',Path(directory)):
            path=Path(directory)/'tmp/hv15-native/short-run.lock';path.parent.mkdir(parents=True)
            with path.open('a') as held:
                runner.fcntl.flock(held,runner.fcntl.LOCK_EX)
                cancel=threading.Event();cancel.set()
                with self.assertRaises(TimeoutError):
                    with runner.experiment_lock(cancel):self.fail('contended experiment lock entered')


if __name__=='__main__':unittest.main()
