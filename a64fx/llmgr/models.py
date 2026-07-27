#!/usr/bin/env python3
"""Model adapters for llmgr.

An adapter knows how to translate a JSON config into the argv/env/cwd for one
of the existing launcher scripts.  llmgr itself stays model-agnostic: it only
supervises processes, so adding a model means adding an adapter here, not
touching llmgr_server.py.

Deliberately thin.  The launcher scripts already encode the hard-won A64FX
environment (OMP_NUM_THREADS=47, XOS_MMM_L_PAGING_POLICY=demand:demand:demand,
and for laguna the *absence* of FLIB_BARRIER=HARD, which would force 48 threads
and ~4x-slow the matvec kernels).  Adapters must NOT re-set those -- an adapter
that "helpfully" exports OMP_NUM_THREADS=48 silently costs ~40% throughput.
Only env the caller explicitly asks for is layered on top.

Standard library only.
"""

import glob
import os

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LAGUNA_DIR = os.path.join(REPO, "a64fx", "laguna-s21")
GEMMA4_DIR = os.path.join(REPO, "a64fx", "gemma4-mn")
UTOFU_DIR = os.path.join(REPO, "a64fx", "utofu-tests")


class ConfigError(ValueError):
    """Bad request config -- reported to the client as HTTP 400."""


def _int(cfg, key, default=None, required=False):
    v = cfg.get(key, default)
    if v is None:
        if required:
            raise ConfigError("missing required field: %s" % key)
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        raise ConfigError("%s must be an integer (got %r)" % (key, v))


def _extra(cfg):
    """Pass-through flags for the runner; both launchers forward unknowns."""
    extra = cfg.get("extra") or []
    if isinstance(extra, str):
        raise ConfigError("extra must be a list of strings, not a string")
    out = []
    for a in extra:
        if not isinstance(a, (str, int, float)):
            raise ConfigError("extra elements must be scalars (got %r)" % (a,))
        out.append(str(a))
    return out


def _env_overrides(cfg):
    env = cfg.get("env") or {}
    if not isinstance(env, dict):
        raise ConfigError("env must be an object")
    return {str(k): str(v) for k, v in env.items()}


class Adapter:
    """Base: subclasses fill in the command builders they support."""

    name = "?"
    variants = ()
    default_variant = None
    supports_serve = False

    def default_np(self):
        return int(os.environ.get("PJM_MPI_PROC", "12"))

    # Each returns (argv, env_overrides, cwd).  Raise ConfigError for bad input,
    # NotImplementedError when the model has no such mode.
    def build(self, cfg):
        raise NotImplementedError("%s has no build mode" % self.name)

    def stage(self, cfg):
        raise NotImplementedError("%s has no stage mode" % self.name)

    def serve(self, cfg):
        raise NotImplementedError("%s has no serve mode" % self.name)

    def generate(self, cfg):
        raise NotImplementedError("%s has no generate mode" % self.name)

    def stage_dir(self, cfg):
        raise NotImplementedError

    def result_path(self, cfg):
        """File a one-shot run leaves its output in, if it has one.

        Runners that print to rank stdout have nothing here; those that write a
        result file do, and llmgr appends it to the child's log on exit so the
        answer is visible without knowing the runner's file conventions.
        """
        return None

    def readiness(self, cfg, since):
        """(ready, detail) for a serve child started at unix time `since`.

        MUST be passive -- read files, never send the runner a request.  An
        HTTP probe is not free here: every accepted connection on rank 0 drives
        a collective across all ranks (see laguna_serve.inc), so probing while
        the slowest rank is still loading its weights runs collectives the
        other ranks never join and desynchronises the whole job.  Measured on
        12 nodes: load times spread 184s..251s while rank 0 began accepting at
        206s, and one-second /health probes left every rank stranded at a
        different sid ("bcast timeout sid=4 want=2 got=1").
        """
        return False, "no readiness check for %s" % self.name

    def runner_bin(self, cfg):
        """Path of the per-rank binary, for fapp-wrapped profiling."""
        raise NotImplementedError("%s cannot be profiled" % self.name)

    def profile_argv(self, cfg):
        """argv for the runner binary itself (no mpiexec, no launcher)."""
        raise NotImplementedError("%s cannot be profiled" % self.name)

    def variant(self, cfg):
        v = cfg.get("variant") or self.default_variant
        if v not in self.variants:
            raise ConfigError("variant must be one of %s (got %r)"
                              % ("|".join(self.variants), v))
        return v


class LagunaAdapter(Adapter):
    """Laguna S-2.1, via a64fx/laguna-s21/run_laguna_s21_12n.sh.

    The only model here with a real server mode: the runner's own --serve HTTP
    port (laguna_serve.inc).  llmgr proxies to it rather than re-implementing
    it, which also keeps the rank-0-collective invariant that file's header
    describes entirely inside the C code where it is already correct.
    """

    name = "laguna"
    variants = ("int4", "bf16", "fp8")
    default_variant = "int4"
    supports_serve = True

    LAUNCHER = os.path.join(LAGUNA_DIR, "run_laguna_s21_12n.sh")

    _RUNNER_BIN = {
        "int4": "laguna_s21_ep_runner",
        "bf16": "laguna_s21_bf16_ep_runner",
        "fp8": "laguna_s21_fp8_ep_runner",
    }
    _DEFAULT_MODEL_DIR = {
        "int4": "~/models/laguna-s21-int4",
        "bf16": "~/models/laguna-s21",
        "fp8": "~/models/laguna-s21-fp8",
    }
    _DEFAULT_NSHARDS = {"int4": 15, "bf16": 46, "fp8": 24}

    def _variant_flag(self, variant):
        return [] if variant == "int4" else ["--%s" % variant]

    def _common(self, cfg, variant):
        """Flags shared by every launcher mode."""
        argv = []
        np_ = _int(cfg, "np", self.default_np())
        argv += ["--np", str(np_)]
        for key, flag in (("model_dir", "--model-dir"),
                          ("stage_dir", "--stage-dir"),
                          ("nshards", "--nshards")):
            if cfg.get(key) is not None:
                argv += [flag, str(cfg[key])]
        return argv

    def stage_dir(self, cfg):
        if cfg.get("stage_dir"):
            return str(cfg["stage_dir"])
        variant = self.variant(cfg)
        np_ = _int(cfg, "np", self.default_np())
        user = os.environ.get("USER", "unknown")
        suffix = {"int4": "", "bf16": "-bf16", "fp8": "-fp8"}[variant]
        return "/local/%s/laguna-s21%s-ep%d" % (user, suffix, np_)

    def model_dir(self, cfg):
        if cfg.get("model_dir"):
            return str(cfg["model_dir"])
        return os.path.expanduser(self._DEFAULT_MODEL_DIR[self.variant(cfg)])

    def build(self, cfg):
        variant = self.variant(cfg)
        targets = ["all"] + ([] if variant == "int4" else [variant])
        argv = ["make", "-C", LAGUNA_DIR] + targets + \
               ["CC=%s" % cfg.get("cc", "fcc"), "OPENMP=1"]
        if cfg.get("clean"):
            # `make clean all` in one invocation is not reliably ordered under
            # -j; run them as an explicit sequence instead.
            argv = ["sh", "-c", "make -C %s clean && %s"
                    % (LAGUNA_DIR, " ".join(argv))]
        return argv, _env_overrides(cfg), LAGUNA_DIR

    def stage(self, cfg):
        variant = self.variant(cfg)
        argv = [self.LAUNCHER, "stage"] + self._variant_flag(variant) \
            + self._common(cfg, variant) + _extra(cfg)
        return argv, _env_overrides(cfg), LAGUNA_DIR

    def serve(self, cfg):
        variant = self.variant(cfg)
        port = _int(cfg, "port", required=True)
        maxpos = _int(cfg, "maxpos", 8192)
        argv = [self.LAUNCHER, "serve", "--port", str(port),
                "--maxpos", str(maxpos)]
        argv += self._variant_flag(variant) + self._common(cfg, variant)
        argv += ["--layers", str(_int(cfg, "layers", 48))]
        if not cfg.get("stage", False):
            argv += ["--no-stage"]
        for key, flag in (("max_batch", "--max-batch"),
                          ("pchunk", "--pchunk")):
            if cfg.get(key) is not None:
                argv += [flag, str(cfg[key])]
        argv += _extra(cfg)
        return argv, _env_overrides(cfg), LAGUNA_DIR

    def generate(self, cfg):
        variant = self.variant(cfg)
        argv = [self.LAUNCHER, "generate"]
        if cfg.get("ids"):
            argv += ["--ids", str(cfg["ids"])]
        elif cfg.get("chat"):
            argv += ["--chat", str(cfg["chat"])]
            if cfg.get("system"):
                argv += ["--system", str(cfg["system"])]
            if cfg.get("no_think"):
                argv += ["--no-think"]
        else:
            argv += ["--prompt", str(cfg.get("prompt", "The capital of France is"))]
        argv += ["--max-new", str(_int(cfg, "max_new", 48)),
                 "--layers", str(_int(cfg, "layers", 48))]
        argv += self._variant_flag(variant) + self._common(cfg, variant)
        if not cfg.get("stage", False):
            argv += ["--no-stage"]
        argv += _extra(cfg)
        return argv, _env_overrides(cfg), LAGUNA_DIR

    # Marker the runner prints once its HTTP listener is up (rank 0 only).
    SERVING_MARKER = "serving HTTP on port"
    LOADED_MARKER = "loaded in"

    def run_dir(self, since):
        """The gen_<timestamp> working directory the launcher just created.

        run_laguna_s21_12n.sh mints one per invocation and every rank writes
        laguna_stderr_rank<NN>.txt into it, which is the only per-rank progress
        signal available without touching the C code.
        """
        best = None
        for d in glob.glob(os.path.join(LAGUNA_DIR, "gen_*")):
            if not os.path.isdir(d):
                continue
            try:
                mtime = os.path.getmtime(d)
            except OSError:
                continue
            # 5s of slack: the directory is created a moment after we fork.
            if mtime >= since - 5 and (best is None or d > best):
                best = d
        return best

    def readiness(self, cfg, since):
        d = self.run_dir(since)
        if d is None:
            return False, "no run dir yet"
        np_ = _int(cfg, "np", self.default_np())
        loaded = 0
        serving = False
        for p in glob.glob(os.path.join(d, "laguna_stderr_rank*.txt")):
            try:
                with open(p, "r", errors="replace") as f:
                    text = f.read()
            except OSError:
                continue
            if self.LOADED_MARKER in text:
                loaded += 1
            if self.SERVING_MARKER in text:
                serving = True
        ok = loaded >= np_ and serving
        return ok, "loaded %d/%d serving=%s dir=%s" % (
            loaded, np_, serving, os.path.basename(d))

    def runner_bin(self, cfg):
        return os.path.join(LAGUNA_DIR, "build",
                            self._RUNNER_BIN[self.variant(cfg)])

    def profile_argv(self, cfg):
        """Runner argv for an fapp-wrapped run: a bounded --generate.

        Profiling must wrap the *per-rank binary*, not the launcher, because
        fapp collects one PMU dataset per process (see
        a64fx/glm5/run_prefill_fapp_12n.sh).  So this bypasses the launcher and
        requires weights already staged plus a tofu_topo.txt in the cwd.
        """
        ids = cfg.get("ids")
        if not ids:
            raise ConfigError("profile needs 'ids' (a pre-tokenized ids file); "
                              "tokenize with tools/laguna_tok.py first")
        # Every rank opens this path, so it must be on the shared filesystem.
        # /tmp and /local are node-local: rank 0 would read it happily and the
        # other 11 would fail to open it.
        ids = str(ids)
        for local in ("/tmp/", "/local/", "/var/tmp/"):
            if ids.startswith(local):
                raise ConfigError(
                    "ids file %s is on node-local storage (%s); every rank must "
                    "be able to open it, so put it on the shared FS (e.g. under "
                    "$HOME or the repo)" % (ids, local.rstrip("/")))
        return ["--generate", "--ids", str(ids),
                "--max-new", str(_int(cfg, "max_new", 16)),
                "--layers", str(_int(cfg, "layers", 48)),
                "--stage-dir", self.stage_dir(cfg),
                "--gen-out", "gen.ids"] + _extra(cfg)


class Gemma4Adapter(Adapter):
    """Gemma-4 12B pipeline-parallel, via a64fx/gemma4-mn/run_gemma4_pp.sh.

    No server mode: the runner takes positional args, generates, writes a
    result file, and exits.  So every gemma4 run is a llmgr `oneshot` child and
    /generate is not available for it -- start a run and read its log.
    """

    name = "gemma4"
    variants = ("pp",)
    default_variant = "pp"
    supports_serve = False

    LAUNCHER = os.path.join(GEMMA4_DIR, "run_gemma4_pp.sh")

    def default_np(self):
        # run_gemma4_pp.sh drops one node from the allocation by default
        # (EXCLUDE=0,0,0) so the login/agent node is not OOM-killed.
        return max(1, int(os.environ.get("PJM_MPI_PROC", "12")) - 1)

    def stage_dir(self, cfg):
        return str(cfg.get("stage_dir") or "/local/gemma4_pp")

    def model_dir(self, cfg):
        return str(cfg.get("gguf") or os.path.expanduser(
            "~/models/gemma4/12b/gemma-4-12b-it-BF16.gguf"))

    def _env(self, cfg, *, skip_stage):
        """run_gemma4_pp.sh is env-driven, not flag-driven."""
        env = {
            "NP": str(_int(cfg, "np", self.default_np())),
            "GGUF": self.model_dir(cfg),
            "STAGE_DIR": self.stage_dir(cfg),
            "MAXGEN": str(_int(cfg, "max_new", 32)),
            "SKIP_STAGE": "1" if skip_stage else "0",
        }
        for key, envname in (("exclude", "EXCLUDE"),
                             ("prompt_ids", "PROMPT_IDS"),
                             ("threads", "LLM_THREADS"),
                             ("max_seq", "MAX_SEQ"),
                             ("result_file", "GEMMA4_RESULT_FILE")):
            if cfg.get(key) is not None:
                env[envname] = str(cfg[key])
        for key, envname in (("q4_int8", "TF_Q4_INT8"),
                             ("lmhead_int8", "TF_LMHEAD_INT8"),
                             ("persist", "GEMMA4_PP_PERSIST"),
                             ("dprof", "TF_DPROF")):
            if cfg.get(key) is not None:
                env[envname] = "1" if cfg[key] else "0"
        env.update(_env_overrides(cfg))
        return env

    def result_path(self, cfg):
        # gemma4_pp_runner.c writes GEMMA4_RESULT_FILE (relative to its cwd,
        # which the launcher leaves as GEMMA4_DIR).
        name = str(cfg.get("result_file") or "gemma4_pp_result.txt")
        return name if os.path.isabs(name) else os.path.join(GEMMA4_DIR, name)

    def build(self, cfg):
        # No Makefile here; run_gemma4_pp.sh compiles inline. Mirror its lines.
        cc = cfg.get("cc", "fcc")
        script = (
            "set -e; cd %(d)s; "
            "make -C %(u)s tofu_topo_helper >/dev/null; "
            "%(cc)s -Nclang -O2 -D_GNU_SOURCE -I../../common "
            "gemma4_stage.c -o gemma4_stage; "
            "%(cc)s -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -fopenmp "
            "-D_GNU_SOURCE -I../../common gemma4_pp_runner.c "
            "-lm -lpthread -lhwb -ltofucom -o gemma4_pp_runner"
        ) % {"d": GEMMA4_DIR, "u": UTOFU_DIR, "cc": cc}
        return ["sh", "-c", script], _env_overrides(cfg), GEMMA4_DIR

    def stage(self, cfg):
        # The launcher stages then runs; there is no stage-only mode, so stage
        # with the smallest possible generation to keep it short.
        env = self._env(cfg, skip_stage=False)
        env["MAXGEN"] = "1"
        return [self.LAUNCHER], env, GEMMA4_DIR

    def generate(self, cfg):
        return [self.LAUNCHER], self._env(cfg, skip_stage=not cfg.get("stage", False)), GEMMA4_DIR

    def runner_bin(self, cfg):
        return os.path.join(GEMMA4_DIR, "gemma4_pp_runner")

    def profile_argv(self, cfg):
        return [self.model_dir(cfg), self.stage_dir(cfg),
                str(cfg.get("prompt_ids", "")), str(_int(cfg, "max_new", 8))]


ADAPTERS = {a.name: a() for a in (LagunaAdapter, Gemma4Adapter)}


def get(model):
    a = ADAPTERS.get(model)
    if a is None:
        raise ConfigError("unknown model %r (have: %s)"
                          % (model, ", ".join(sorted(ADAPTERS))))
    return a


def describe():
    return {
        name: {
            "variants": list(a.variants),
            "default_variant": a.default_variant,
            "supports_serve": a.supports_serve,
            "default_np": a.default_np(),
        }
        for name, a in ADAPTERS.items()
    }
