"""Run both complete quality pipelines and independent references, with receipts."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import signal
import threading
import time

from generate import Cancelled, ROOT, RUNNER, atomic_json, digest, generate, run_process

VERIFIER = ROOT / "ref/hunyuan_video15_native/verify.py"
LOCK = ROOT / "tmp/pixal3d/device-locks/cuda-0.lock"
PROMPT = "The same person smiles gently, then relaxes. Fixed frontal camera."


def normalize_paths(args):
    for key in ("model", "image", "out", "reference_python", "runner", "adopt_run", "adopt_captures"):
        value = getattr(args, key)
        if value is not None:
            # Resolving a venv's interpreter symlink bypasses its pyvenv.cfg.
            setattr(args, key, Path(os.path.abspath(value)) if key == "reference_python" else value.resolve())


@contextmanager
def device_lock(cancel, path=LOCK):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as lock:
        while True:
            if cancel.is_set():
                raise Cancelled("cancelled while waiting for GPU")
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                cancel.wait(.25)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def validate_generation(run, task, args, runner_hash):
    manifest = json.loads((run / "manifest.json").read_text())
    expected = dict(backend="hv15n_cuda_experimental", task=task, preset="quality", prompt=args.prompt,
                    negative_prompt=args.negative_prompt, seed=args.seed,
                    steps=50, cfg=6, flow_shift=5, frames=81, fps=24,
                    width=480, height=848, gemm="repo", gemm_fallback="error",
                    runner_sha256=runner_hash)
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError("completed generation differs: " + key)
    if manifest.get("model") != json.loads((args.model / "model.json").read_text()):
        raise ValueError("completed generation model receipt differs")
    metrics = manifest.get("metrics", {})
    if (metrics.get("backend") != "hv15n_cuda" or metrics.get("memory_fit") != "pass"
            or metrics.get("repo_gemm_calls", 0) <= 0
            or metrics.get("cublas_gemm_calls") != 0 or metrics.get("fallback_gemm_calls") != 0):
        raise ValueError("completed generation lacks strict GEMM/memory evidence")
    if task == "i2v" and manifest.get("image_sha256") != digest(args.image):
        raise ValueError("completed generation portrait differs")
    if not (run / "clip.mp4").is_file():
        raise ValueError("completed generation clip missing")
    return manifest


def process_identity(pid, run):
    """Bind adoption to this exact generator/output and a Linux process start time."""
    proc = Path("/proc") / str(pid)
    command = (proc / "cmdline").read_bytes().decode().rstrip("\0").split("\0")
    if not any(Path(word).name == "generate.py" for word in command) or "--out" not in command:
        raise ValueError("adopted PID is not the generator")
    cwd = (proc / "cwd").resolve()
    output = Path(command[command.index("--out") + 1])
    if (cwd / output).resolve() != run.resolve():
        raise ValueError("adopted PID writes another run")
    return (proc / "stat").read_text().rsplit(")", 1)[1].split()[19]


def wait_generation(pid, run, cancel):
    identity = process_identity(pid, run)
    while not (run / "manifest.json").is_file():
        if cancel.is_set():
            # SIGINT gives the original wrapper its normal finally/cleanup path.
            try:
                if process_identity(pid, run) == identity:
                    os.kill(pid, signal.SIGINT)
            except (FileNotFoundError, ProcessLookupError):
                pass
            raise Cancelled("cancelled adopted generation")
        try:
            if process_identity(pid, run) != identity:
                raise ValueError("adopted process changed before completing its run")
        except FileNotFoundError as error:
            raise RuntimeError("adopted generation exited without a complete manifest") from error
        cancel.wait(.5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "image", "out", "reference-python"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--runner", type=Path, default=RUNNER)
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--adopt-run", type=Path)
    parser.add_argument("--adopt-captures", type=Path)
    parser.add_argument("--adopt-pid", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if bool(args.adopt_run) != bool(args.adopt_captures) or (args.adopt_pid and not args.adopt_run):
        parser.error("adoption needs both --adopt-run and --adopt-captures")
    normalize_paths(args)
    if args.out == args.model or args.out in args.model.parents or args.model in args.out.parents:
        parser.error("campaign and model directories must be disjoint")
    cancel = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda unused, unused_frame: cancel.set())
    args.out.mkdir(parents=True, exist_ok=True)
    scratch = args.out / "scratch"
    scratch.mkdir(exist_ok=True)
    os.environ["TMPDIR"] = str(scratch)
    state_path = args.out / "campaign.json"
    identity = dict(model=str(args.model), model_manifest_sha256=digest(args.model / "model.json"),
                    image=str(args.image), image_sha256=digest(args.image), prompt=args.prompt,
                    negative_prompt=args.negative_prompt, seed=args.seed, runner=str(args.runner),
                    runner_sha256=digest(args.runner), verifier_sha256=digest(VERIFIER),
                    reference_python=str(args.reference_python),
                    adopt_run=str(args.adopt_run) if args.adopt_run else None,
                    adopt_captures=str(args.adopt_captures) if args.adopt_captures else None)
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if not args.resume or state.get("identity") != identity:
            raise ValueError("existing campaign needs --resume and identical inputs/binaries")
    else:
        state = dict(schema="hv15n.quality_campaign.v1", identity=identity, tasks={})

    def update(**values):
        state.update(values, controller_pid=os.getpid(), updated_unix=time.time())
        atomic_json(state_path, state)

    def check_sources():
        if digest(VERIFIER) != identity["verifier_sha256"] or digest(args.runner) != identity["runner_sha256"]:
            raise ValueError("runner or reference verifier changed during the campaign")

    def reference_phase(task, run, captures, out, phase, components=None):
        check_sources()
        command = [args.reference_python, VERIFIER, "--model", args.model,
                   "--actual", captures, "--out", out, "--generation-manifest", run / "manifest.json",
                   "--phase", phase]
        if components:
            command += ["--components", *components]
        if task == "i2v":
            command += ["--image", run / "input.png", "--vision-pixels", run / "vision_pixels.f32"]
        gpu = phase in ("denoise", "decode") or components == ["vae_encoded"]
        label = "vae_encode" if components == ["vae_encoded"] else phase
        update(status="running", task=task, phase=label, command=list(map(str, command)))
        print(f"PHASE {task} {label}", flush=True)
        with (out.parent / (label + ".log")).open("w") as log:
            def execute():
                run_process(command, cancel=cancel, log=log,
                            on_start=lambda pid: update(reference_pid=pid))
            if gpu:
                with device_lock(cancel):
                    execute()
            else:
                execute()

    try:
        update(status="running", pass_both_quality_pipelines=False, error=None)
        for task in ("i2v", "t2v"):
            task_dir = args.out / task
            task_dir.mkdir(exist_ok=True)
            run, captures = task_dir / "run", task_dir / "captures"
            if task == "i2v" and args.adopt_run:
                run, captures = args.adopt_run, args.adopt_captures
            check_sources()
            state["tasks"][task] = dict(status="running", run=str(run), captures=str(captures))
            update(task=task, phase="generation", run=str(run), captures=str(captures))
            if not (run / "manifest.json").exists():
                if task == "i2v" and args.adopt_run:
                    if not args.adopt_pid:
                        raise ValueError("unfinished adopted run needs its live generator PID")
                    update(adopted_generator_pid=args.adopt_pid)
                    wait_generation(args.adopt_pid, run, cancel)
                else:
                    with device_lock(cancel):
                        generate(model=args.model, out=run, task=task, preset="quality", prompt=args.prompt,
                                 negative_prompt=args.negative_prompt, image=args.image if task == "i2v" else None,
                                 seed=args.seed, gemm="repo", gemm_fallback="error", runner=args.runner,
                                 allow_experimental=True, dump_dir=captures, keep_frames=True, cancel=cancel,
                                 progress=lambda step, total: update(step=step, total=total))
            validate_generation(run, task, args, identity["runner_sha256"])
            reference = task_dir / "reference"
            reference.mkdir(exist_ok=True)
            if not (reference / "compare_parity.json").is_file():
                components = ["qwen_hidden", "qwen_negative_hidden", "byt5_hidden"]
                if task == "i2v":
                    components += ["siglip_hidden"]
                reference_phase(task, run, captures, reference, "encoders", components)
                if task == "i2v":
                    reference_phase(task, run, captures, reference, "encoders", ["vae_encoded"])
                reference_phase(task, run, captures, reference, "denoise")
                reference_phase(task, run, captures, reference, "decode")
            # Repeat the fail-closed final gate on resume; never trust a saved pass alone.
            reference_phase(task, run, captures, reference, "compare")
            report = json.loads((reference / "compare_parity.json").read_text())
            if not report.get("pass") or report.get("scope") != "pipeline":
                raise ValueError("complete quality pipeline did not pass")
            state["tasks"][task] = dict(status="pass", manifest_sha256=digest(run / "manifest.json"),
                                        report_sha256=digest(reference / "compare_parity.json"),
                                        report=str(reference / "compare_parity.json"))
            update()
        update(status="complete", phase="complete", pass_both_quality_pipelines=True)
    except BaseException as error:
        if state.get("task") in state["tasks"]:
            state["tasks"][state["task"]].update(status="cancelled" if isinstance(error, Cancelled) else "failed",
                                                error=str(error))
        update(status="cancelled" if isinstance(error, Cancelled) else "failed", error=str(error),
               pass_both_quality_pipelines=False)
        raise


if __name__ == "__main__":
    main()
