"""Stream worker output while keeping cancellation independent of pipe EOF."""
import os
import queue
import signal
import subprocess
import threading

from . import gpu


def _stop_group(proc):
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            pass
    except ProcessLookupError:
        pass
    finally:
        # Descendants can retain stdout after their parent has already exited.
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()


def run(command, cancel, on_line, *, cwd, env):
    if cancel.is_set():
        raise gpu.Cancelled('cancelled')
    proc = subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, start_new_session=True)
    lines = queue.Queue(maxsize=256)
    stopping = threading.Event()

    def enqueue(value):
        while not stopping.is_set():
            try:
                lines.put(value, timeout=.1)
                return
            except queue.Full:
                pass

    def drain():
        try:
            for line in proc.stdout:
                enqueue(line)
        except Exception as exc:
            enqueue(exc)
        finally:
            enqueue(None)

    reader = threading.Thread(target=drain, daemon=True)
    reader.start()
    try:
        while True:
            if cancel.is_set():
                raise gpu.Cancelled('cancelled')
            try:
                line = lines.get(timeout=.1)
            except queue.Empty:
                continue
            if line is None:
                break
            if isinstance(line, Exception):
                raise line
            on_line(line.rstrip())
        while proc.poll() is None:
            if cancel.wait(.1):
                raise gpu.Cancelled('cancelled')
        return proc.wait()
    except BaseException:
        _stop_group(proc)
        raise
    finally:
        stopping.set()
        reader.join()
        proc.stdout.close()
