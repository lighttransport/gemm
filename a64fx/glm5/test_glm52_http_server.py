#!/usr/bin/env python3
import argparse
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import glm52_http_server as http


class ServiceTest(unittest.TestCase):
    def test_persistent_worker_protocol_and_shutdown(self):
        with tempfile.TemporaryDirectory() as td:
            runner = Path(td) / "fake_runner.py"
            runner.write_text("""#!/usr/bin/env python3
import os, sys, time
from pathlib import Path
d=Path(sys.argv[sys.argv.index('--serve-dir')+1]); d.mkdir(exist_ok=True)
(d/'ready').write_text('ready\\n'); last=0
while not (d/'stop').exists():
    r=d/'request'
    if r.exists():
        f=r.read_text().split(); seq=int(f[0])
        if seq>last:
            n=int(f[2]); prefix=f[4]
            for i in range(n): Path(prefix+'_%03d.txt'%i).write_text('%d\\n'%(100+i))
            (d/'done.tmp').write_text('%d 0\\n'%seq); os.replace(str(d/'done.tmp'),str(d/'done')); last=seq
    time.sleep(.01)
""")
            runner.chmod(0o755)
            args = argparse.Namespace(
                runner=str(runner), work_dir=td, max_context=128, max_tokens=8,
                max_slots=2, startup_timeout=5, worker_pchunk=64,
            )
            worker = http.PersistentWorker(args)
            try:
                self.assertEqual(worker.submit([[1], [2]], 2, 5), [[100], [101]])
            finally:
                worker.stop()
            self.assertIsNotNone(worker.proc.poll())

    def test_completion_contract_and_long_context_override(self):
        with tempfile.TemporaryDirectory() as td:
            args = argparse.Namespace(
                tokenizer=http.TOKJSON, max_tokens=8, max_context=262144,
                work_dir=td, runner="fake-runner", int4_threshold=1, timeout=30,
                max_body=1024,
                max_slots=4, max_total_context=262144,
            )
            service = http.Service(args)

            def fake_run(cmd, **kwargs):
                out = Path(cmd[cmd.index("--gen-out") + 1])
                out.write_text("198 200\n")
                self.assertIn("--stable-outputs", cmd)
                self.assertIn("--kv-tier-bf16=0", cmd)
                return type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})()

            with patch.object(http.subprocess, "run", side_effect=fake_run):
                result = service.complete({"prompt": "hello", "max_tokens": 2})
            self.assertEqual(result["object"], "text_completion")
            self.assertEqual(result["usage"]["completion_tokens"], 2)
            self.assertEqual(result["choices"][0]["index"], 0)

    def test_rejects_invalid_limits(self):
        args = argparse.Namespace(
            tokenizer=http.TOKJSON, max_tokens=8, max_context=16,
            work_dir="/tmp", runner="unused", int4_threshold=23000,
            timeout=30, max_body=1024,
            max_slots=4, max_total_context=16,
        )
        service = http.Service(args)
        with self.assertRaises(ValueError):
            service.complete({"prompt": "hello", "max_tokens": 9})

    def test_multiple_independent_context_outputs(self):
        with tempfile.TemporaryDirectory() as td:
            args = argparse.Namespace(
                tokenizer=http.TOKJSON, max_tokens=8, max_context=128,
                max_total_context=256, max_slots=2, work_dir=td,
                runner="fake-runner", int4_threshold=23000, timeout=30, max_body=1024,
            )
            service = http.Service(args)

            def fake_run(cmd, **kwargs):
                prefix = cmd[cmd.index("--out-prefix") + 1]
                Path(prefix + "_000.txt").write_text("9227 22202\n")
                Path(prefix + "_001.txt").write_text("10397 154827\n")
                self.assertEqual(cmd[cmd.index("--slots") + 1], "2")
                return type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})()

            with patch.object(http.subprocess, "run", side_effect=fake_run):
                result = service.complete_many({"prompts": ["one", "two"], "max_tokens": 2})
            self.assertEqual(result["contexts"], 2)
            self.assertEqual([x["index"] for x in result["choices"]], [0, 1])


if __name__ == "__main__":
    unittest.main()
