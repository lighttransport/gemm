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
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LAGUNA_DIR = os.path.join(REPO, "a64fx", "laguna-s21")
GEMMA4_DIR = os.path.join(REPO, "a64fx", "gemma4-mn")
K3_DIR = os.path.join(REPO, "a64fx", "k3")
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
    """Generic runner contract used by llmgr.

    Adapters only translate a model configuration into a command, environment,
    and working directory.  The supervisor owns process groups, logs, rank
    output collection, readiness, stopping, and HTTP proxying.
    """

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

    def launch(self, mode, cfg):
        """Return ``(argv, env, cwd)`` for a standard llmgr operation.

        Keeping this dispatch here lets the supervisor remain generic when a
        new runner adds a mode; the server does not need another model branch.
        """
        builders = {
            "build": self.build,
            "stage": self.stage,
            "serve": self.serve,
            "generate": self.generate,
        }
        try:
            return builders[mode](cfg)
        except KeyError:
            raise ConfigError("unsupported runner operation %r" % mode)

    def variant(self, cfg):
        v = cfg.get("variant") or self.default_variant
        if v not in self.variants:
            raise ConfigError("variant must be one of %s (got %r)"
                              % ("|".join(self.variants), v))
        return v

    def contract(self):
        """Machine-readable capabilities for clients and documentation."""
        modes = []
        for mode in ("build", "stage", "serve", "generate", "profile"):
            if mode == "profile":
                fn = self.profile_argv
            else:
                fn = getattr(self, mode)
            if fn.__func__ is not getattr(Adapter, mode, None):
                modes.append(mode)
        return {"modes": modes, "runner_contract": "llmgr.v1"}


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
    _FP16_KV_RUNNER_BIN = "laguna_s21_fp8_kvfp16_ep_runner"
    _DEFAULT_MODEL_DIR = {
        "int4": "~/models/laguna-s21-int4",
        "bf16": "~/models/laguna-s21",
        "fp8": "~/models/laguna-s21-fp8",
    }
    _DEFAULT_NSHARDS = {"int4": 15, "bf16": 46, "fp8": 24}

    def _variant_flag(self, variant):
        return [] if variant == "int4" else ["--%s" % variant]

    def _kv_flags(self, cfg, variant):
        if not cfg.get("kv_fp16"):
            return []
        if variant != "fp8":
            raise ConfigError("kv_fp16 currently requires variant=fp8")
        return ["--kv-fp16"]

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
        if self._kv_flags(cfg, variant):
            targets.append("fp8-kvfp16")
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
        argv += self._kv_flags(cfg, variant)
        argv += ["--layers", str(_int(cfg, "layers", 48))]
        if not cfg.get("stage", False):
            argv += ["--no-stage"]
        for key, flag in (("max_batch", "--max-batch"),
                          ("pchunk", "--pchunk"),
                          ("ar_groups", "--ar-groups"),
                          ("comm_robust", "--comm-robust"),
                          ("comm_poll_spins", "--comm-poll-spins")):
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
        argv += self._kv_flags(cfg, variant)
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
        variant = self.variant(cfg)
        if self._kv_flags(cfg, variant):
            return os.path.join(LAGUNA_DIR, "build", self._FP16_KV_RUNNER_BIN)
        return os.path.join(LAGUNA_DIR, "build",
                            self._RUNNER_BIN[variant])

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
    """Gemma-4 12B PP/TP via the a64fx/gemma4-mn launchers.

    No server mode: the runner takes positional args, generates, writes a
    result file, and exits.  So every gemma4 run is a llmgr `oneshot` child and
    /generate is not available for it -- start a run and read its log.
    """

    name = "gemma4"
    variants = ("pp", "tp")
    default_variant = "tp"
    supports_serve = False

    LAUNCHERS = {
        "pp": os.path.join(GEMMA4_DIR, "run_gemma4_pp.sh"),
        "tp": os.path.join(GEMMA4_DIR, "run_gemma4_tp.sh"),
    }

    def default_np(self):
        # The batch allocation supplies the usable rank count explicitly. Keep
        # the historical controller-node reserve for callers without --np.
        return max(1, int(os.environ.get("PJM_MPI_PROC", "12")) - 1)

    def _variant(self, cfg):
        return self.variant(cfg)

    def stage_dir(self, cfg):
        default = "/local/gemma4_tp" if self._variant(cfg) == "tp" else "/local/gemma4_pp"
        return str(cfg.get("stage_dir") or default)

    def model_dir(self, cfg):
        return str(cfg.get("gguf") or os.path.expanduser(
            "~/models/gemma4/12b/gemma-4-12b-it-BF16.gguf"))

    def _env(self, cfg, *, skip_stage):
        """Gemma4 launchers are env-driven, not flag-driven."""
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
        if self._variant(cfg) == "tp":
            for key, envname in (("mtp", "GEMMA4_TP_MTP"),
                                 ("spec_k", "GEMMA4_TP_SPEC_K"),
                                 ("batch", "GEMMA4_TP_BATCH")):
                if cfg.get(key) is not None:
                    env[envname] = str(cfg[key])
            if cfg.get("tp_skip_ar") is not None:
                env["TP_SKIP_AR"] = "1" if cfg["tp_skip_ar"] else "0"
        env.update(_env_overrides(cfg))
        return env

    def result_path(self, cfg):
        # Both runners write GEMMA4_RESULT_FILE relative to GEMMA4_DIR.
        default = "gemma4_tp_result.txt" if self._variant(cfg) == "tp" else "gemma4_pp_result.txt"
        name = str(cfg.get("result_file") or default)
        return name if os.path.isabs(name) else os.path.join(GEMMA4_DIR, name)

    def build(self, cfg):
        # No Makefile here; the selected launcher compiles inline.
        cc = cfg.get("cc", "fcc")
        runner = "gemma4_tp_runner.c" if self._variant(cfg) == "tp" else "gemma4_pp_runner.c"
        output = "gemma4_tp_runner" if self._variant(cfg) == "tp" else "gemma4_pp_runner"
        script = (
            "set -e; cd %(d)s; "
            "make -C %(u)s tofu_topo_helper >/dev/null; "
            "%(cc)s -Nclang -O2 -D_GNU_SOURCE -I../../common "
            "gemma4_stage.c -o gemma4_stage; "
            "%(cc)s -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -fopenmp "
            "-D_GNU_SOURCE -I../../common %(runner)s "
            "-lm -lpthread -lhwb -ltofucom -o %(output)s"
        ) % {"d": GEMMA4_DIR, "u": UTOFU_DIR, "cc": cc,
              "runner": runner, "output": output}
        return ["sh", "-c", script], _env_overrides(cfg), GEMMA4_DIR

    def stage(self, cfg):
        # The launcher stages then runs; there is no stage-only mode, so stage
        # with the smallest possible generation to keep it short.
        env = self._env(cfg, skip_stage=False)
        env["MAXGEN"] = "1"
        return [self.LAUNCHERS[self._variant(cfg)]], env, GEMMA4_DIR

    def generate(self, cfg):
        return [self.LAUNCHERS[self._variant(cfg)]], self._env(
            cfg, skip_stage=not cfg.get("stage", False)), GEMMA4_DIR

    def runner_bin(self, cfg):
        name = "gemma4_tp_runner" if self._variant(cfg) == "tp" else "gemma4_pp_runner"
        return os.path.join(GEMMA4_DIR, name)

    def profile_argv(self, cfg):
        return [self.model_dir(cfg), self.stage_dir(cfg),
                str(cfg.get("prompt_ids", "")), str(_int(cfg, "max_new", 8))]


class K3Adapter(Adapter):
    """Kimi K3 TP partial runner with real MXFP4 expert slices.

    K3 does not yet own a tokenizer, embedding, complete dense path, or LM
    head, so this is deliberately a one-shot measured-kernel contract.  llmgr
    supplies the HTTP control plane (build/stage/start/stop/log and /bash), but
    does not advertise a semantic serving endpoint that the runner cannot
    implement honestly.
    """

    name = "k3"
    variants = ("partial",)
    default_variant = "partial"
    supports_serve = False
    LAUNCHER = os.path.join(K3_DIR, "run_k3_ep.sh")

    def _np(self, cfg):
        np_ = _int(cfg, "np", self.default_np())
        if np_ < 1 or np_ > 96:
            raise ConfigError("np must be in [1,96] for K3 (got %d)" % np_)
        return np_

    def stage_dir(self, cfg):
        if cfg.get("stage_dir"):
            return str(cfg["stage_dir"])
        user = os.environ.get("USER", "unknown")
        job = os.environ.get("PJM_JOBID", "interactive")
        return "/local/%s/k3-llmgr-%s" % (user, job)

    def model_dir(self, cfg):
        return str(cfg.get("model_dir") or os.path.expanduser("~/models/kimi-k3"))

    def _result_dir(self, cfg, operation):
        if cfg.get("result_dir"):
            return str(cfg["result_dir"])
        job = os.environ.get("PJM_JOBID", "interactive")
        stamp = int(time.time() * 1000000)
        return os.path.join(K3_DIR, "logs", "llmgr-%s-%s-%d" %
                            (operation, job, stamp))

    def _runner_flags(self, cfg):
        np_ = self._np(cfg)
        layer = _int(cfg, "layer", 1)
        layers = _int(cfg, "layers", 1)
        token_default = cfg.get("max_new")
        tokens = _int(cfg, "tokens", 256 if token_default is None else token_default)
        threads = _int(cfg, "threads", 48)
        heartbeat = _int(cfg, "heartbeat_tokens", 1024)
        min_available = _int(cfg, "min_available_mib", 2048)
        if not 0 <= layer <= 92:
            raise ConfigError("layer must be in [0,92]")
        if not 1 <= layers <= 93 or layer + layers > 93:
            raise ConfigError("invalid K3 layer range [%d,%d)" %
                              (layer, layer + layers))
        if not 1 <= tokens <= 1048576:
            raise ConfigError("tokens must be in [1,1048576]")
        if not 1 <= threads <= 48:
            raise ConfigError("threads must be in [1,48]")
        if heartbeat is None or not 0 <= heartbeat <= 1048576:
            raise ConfigError("heartbeat_tokens must be in [0,1048576]")
        if min_available is None or not 0 <= min_available <= 1048576:
            raise ConfigError("min_available_mib must be in [0,1048576]")
        argv = ["--nodes", str(np_), "--layer", str(layer),
                "--layers", str(layers), "--tokens", str(tokens),
                "--threads", str(threads),
                "--heartbeat-tokens", str(heartbeat),
                "--min-available-mib", str(min_available),
                "--ar-groups", str(cfg.get("ar_groups", "auto"))]
        if cfg.get("profile", True):
            argv.append("--profile")
        return argv

    def build(self, cfg):
        if cfg.get("clean"):
            argv = ["sh", "-c", "make -C %s clean && make -C %s runner k3_moe_probe"
                    % (K3_DIR, K3_DIR)]
        else:
            argv = ["make", "-C", K3_DIR, "runner", "k3_moe_probe"]
        return argv, _env_overrides(cfg), K3_DIR

    def stage(self, cfg):
        layer = _int(cfg, "layer", 1)
        experts = str(cfg.get("experts", "0-15"))
        argv = [self.LAUNCHER, "--mode", "real", "--stage-only",
                "--nodes", str(self._np(cfg)), "--layer", str(layer),
                "--layers", "1", "--tokens", "1", "--experts", experts,
                "--model-dir", self.model_dir(cfg),
                "--stage-dir", self.stage_dir(cfg),
                "--result-dir", self._result_dir(cfg, "stage")]
        argv += _extra(cfg)
        return argv, _env_overrides(cfg), K3_DIR

    def generate(self, cfg):
        real = not cfg.get("dummy", False)
        argv = [self.LAUNCHER, "--mode", "real" if real else "dummy"]
        argv += self._runner_flags(cfg)
        if real:
            if _int(cfg, "layer", 1) == 0 or _int(cfg, "layers", 1) != 1:
                raise ConfigError("K3 real partial mode requires layer 1..92 and layers=1")
            argv += ["--experts", str(cfg.get("experts", "0-15")),
                     "--model-dir", self.model_dir(cfg),
                     "--stage-dir", self.stage_dir(cfg)]
            if not cfg.get("stage", False):
                argv.append("--reuse-stage")
        argv += ["--result-dir", self._result_dir(cfg, "run")]
        argv += _extra(cfg)
        return argv, _env_overrides(cfg), K3_DIR

    def runner_bin(self, cfg):
        return os.path.join(K3_DIR, "k3_ep_runner")

    def profile_argv(self, cfg):
        layer = _int(cfg, "layer", 1)
        if layer < 1 or layer > 92:
            raise ConfigError("K3 profile requires layer in [1,92]")
        token_default = cfg.get("max_new")
        tokens = _int(cfg, "tokens", 64 if token_default is None else token_default)
        return ["--mode", "real", "--nodes", str(self._np(cfg)),
                "--layer", str(layer), "--layers", "1",
                "--tokens", str(tokens),
                "--threads", str(_int(cfg, "threads", 48)),
                "--stage-dir", self.stage_dir(cfg), "--status-dir", ".",
                "--topo", "tofu_topo.txt", "--profile"] + _extra(cfg)


ADAPTERS = {a.name: a() for a in (LagunaAdapter, Gemma4Adapter, K3Adapter)}


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
            **a.contract(),
        }
        for name, a in ADAPTERS.items()
    }
