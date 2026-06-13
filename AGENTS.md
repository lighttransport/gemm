# Repository Guidelines

## Project Structure & Module Organization
This repository is organized by hardware backend. Use top-level folders as module boundaries:
- `a64fx/`: A64FX SVE kernels, microbenchmarks, and optimization notes.
- `ryzen/` and `zen2/`: x86_64 AVX2/FMA GEMM, attention, and NN ops.
- `cuda/int8` and `cuda/fp8`: CUDA/cuBLAS experiments and reference tools.
- `vulkan/`: Vulkan compute runners, shaders, and multimodal tests.
- `common/`: shared single-file components (`gguf_loader`, tokenizer, transformer) and `test_*.c` programs.

Keep generated binaries and temporary logs local; do not mix unrelated architecture work in one change.

## Build, Test, and Development Commands
Run commands from repo root unless noted:
- `make -C ryzen` builds AVX2 benchmarks; `make -C ryzen gemm|flash|nn` builds and runs specific suites.
- `make -C zen2` builds the Zen2 benchmark driver.
- `make -C cuda/int8 test` runs INT8 attention/FFN correctness checks.
- `cmake -S vulkan -B vulkan/build && cmake --build vulkan/build -j` builds Vulkan tools and compiles shaders.
- `./vulkan/build/test_vision_encoder` (or `test_vision_multimodal`) runs Vulkan-side validation.
- `make -C a64fx/int8-new 5x4 COMPILER=fcc` builds an A64FX target; use `make -C <dir> clean` to reset artifacts.

## Coding Style & Naming Conventions
- Languages: C, C++, and architecture-specific `.S` assembly.
- Use 4-space indentation and keep brace/style conventions consistent with nearby files.
- Use `snake_case` for functions/files, `UPPER_CASE` for macros/constants.
- Follow existing naming suffixes for backend/type (`_avx2`, `_sve`, `_f32`) and prefixes (`bench_`, `test_`).
- Keep warning-clean builds (`-Wall -Wextra -Wpedantic`) where Makefiles already enforce them.

## Testing Guidelines
There is no single global test harness. Each module owns executable tests/benchmarks.
- Add or update `test_*.c` when changing math kernels or loaders.
- Validate correctness against existing naive/reference paths before reporting performance.
- In PRs, include exact commands run and representative output for both correctness and performance.

## Commit & Pull Request Guidelines
Git history favors short imperative subjects (for example: `Add ...`, `Optimize ...`, `Fix ...`).
- Keep each commit focused on one subsystem/backend.
- PRs should include: purpose, affected hardware/compiler settings, commands executed, and before/after metrics.
- Link related issues and call out any portability limits (ISA, GPU extension, or SDK requirements).

## Pushing & Authorization (agents)
Automated agents working in this tree must NOT run `git push` to any remote without explicit
per-action user permission in the current request.

- Commit freely once a coherent unit of work is done (or when the user says "commit"); always
  report the commit hash and a short diff summary so the user can review before authorizing a
  push.
- Do NOT chain `git push` after a commit in the same shell invocation. Each push needs its
  own explicit request.
- Prior "push" authorization does NOT carry across subsequent commits in the same session.
  One push verb from the user = one push. The next commit re-arms the question.
- The only standing exceptions are explicit push verbs in the user's current-turn message:
  "push it", "git push", "ship to main", "merge + push", etc.
- When unsure, default to NOT pushing. Ask first.

This rule applies to all remotes (origin and any others) and all branches, including feature
branches. It is the same intent as Claude Code's general guidance ("for actions that are hard
to reverse or outward-facing, confirm first; approval in one context doesn't extend to the
next").

### Repo-specific: git runs over SSH on Fugaku
The local working dirs are Mutagen mirrors with no usable `.git` (the gitfile points to a
remote-only worktree). The real repo — worktree branch `ds4p` — lives on Fugaku, so every git
command runs as a chained one-liner: `ssh fugaku 'cd ~/work/gemm/ds4p && git ...'`.

- Edit files locally (Mutagen syncs them to the remote in ~1s), then `git add` / `git commit`
  within a single `ssh fugaku '...'` invocation. Verify the sync landed (e.g. `grep` the change
  on the remote) before staging.
- The "no push chained after a commit in the same shell invocation" rule above is **literal**
  here: never put `git push` inside the same `ssh fugaku '...'` command as a commit. An
  authorized push is always its own separate, explicitly-requested `ssh fugaku '... git push ...'`
  invocation.
- Commit on `ds4p` (the active worktree branch) directly. Pushes target the shared
  `github.com:lighttransport/gemm`, so they always require a current-turn push verb.

### Mandatory pre-push audit checklist

Pushing rewrites shared state on the public remote `github.com:lighttransport/gemm`. Before
**any** `git push` (regular, force, or `--force-with-lease`), complete every step below —
skipping one is a defect. Once a large binary or a leaked credential lands on the remote it is
forever in the public Git history.

All commands below run on Fugaku. Open an interactive remote shell, or wrap each block in a
heredoc to dodge nested-quote pain:

```bash
ssh fugaku 'cd ~/work/gemm/ds4p && bash -s' <<'EOF'
  # ... command block ...
EOF
```

#### 1. Audit the exact commit range about to leave the machine

```bash
# Regular push: range vs the branch's upstream. Force/lease: vs the remote tip.
# ds4p often has no upstream set — fall back to origin/main..HEAD (or the merge-base
# with whatever branch you are pushing to).
RANGE="$(git rev-parse --abbrev-ref --symbolic-full-name @{upstream} 2>/dev/null || echo origin/main)..HEAD"
git log --oneline "$RANGE"
```

Run all four checks against **every** commit in `$RANGE`, not just `HEAD`. Any single offending
commit blocks the push; resolve with `git rebase` (drop/edit — note some sandboxes block `-i`)
or `git filter-repo` (path removal) before continuing.

#### Check 1 — Credentials & sensitive data

Never push a commit that contains, or ever contained within `$RANGE`:

- API keys, **bearer tokens** (notably `DS4P_BASH_HTTP_TOKEN` — must be sourced from the env,
  never a literal), OAuth secrets, AWS/GCP/Azure access keys, SSH/PGP private keys, `.netrc`,
  `.env`, `.npmrc` with `_authToken`.
- Email/password pairs, JWTs, Slack/Discord/GitHub webhook URLs.
- The bash-over-http runtime files `tools/bash_http_job/runtime.*.env` (token-bearing; gitignored —
  confirm none were force-added).
- Internal hosts/addresses: Fugaku nodes (`fn01sv*`, `login*.fugaku.r-ccs.riken.jp`, compute IDs
  like `g30-1100c`), private `10.*` IPs, VPN configs.
- RIKEN allocation & personal paths baked into source/comments: `/vol0006/...`, the project code
  `hp250467`, `/home/<user>/...`, `~/models`, `~/data` — embarrassing rather than dangerous, but
  still stripped.

```bash
git diff "$RANGE" -- ':!**/*.md' ':!**/*.txt' \
  | grep -nIE 'AKIA[0-9A-Z]{16}|ASIA[0-9A-Z]{16}|AIza[0-9A-Za-z_-]{35}|(api[_-]?key|secret|token|password|passwd|bearer|private[_-]?key)[[:space:]]*[:=]|ds4p-bash-http|/vol0006/|hp250467|/home/[a-z]' \
  || echo "No credential-shaped strings found in $RANGE"
```

Heuristic only. If a commit touches anything that *talks* to an external service (the bash-http
bridge, CI, an asset server), eyeball the diff manually too. `.md`/`.txt` are excluded above —
the docs legitimately mention `/tmp/ds4p-bash-http` (a dir name) and Fugaku hostnames; scan
those by eye rather than letting the regex block on them.

##### Automated secret scanners (preferred over the grep heuristic when installed)

When `gitleaks` / `trufflehog` are available, run them over the **same range** as an
entropy- and rule-aware complement. **Any finding blocks the push** (resolve, treat the secret
as compromised, rotate first). If a tool is missing, note the skip and fall back to the grep.

```bash
BASE="$(git rev-parse "${RANGE%%..*}")"
REPO="$(git rev-parse --show-toplevel)"
BRANCH="$(git branch --show-current)"   # ds4p
# Our .git is a *worktree* gitfile, which trufflehog's go-git backend cannot read.
# Point it at the MAIN worktree dir (its .git is a real directory) — on Fugaku this
# resolves to ~/work/gemm/main.
MAIN_WT="$(git worktree list --porcelain | awk '/^worktree /{print $2; exit}')"

if command -v gitleaks >/dev/null 2>&1; then
  gitleaks detect --source "$REPO" --redact --log-opts="$RANGE" \
    && echo "gitleaks: clean over $RANGE" \
    || { echo "ABORT: gitleaks flagged secrets in $RANGE"; exit 1; }
else
  echo "gitleaks not installed; skipping. Rely on the grep heuristic above."
fi

TRUFFLEHOG="${TRUFFLEHOG:-$HOME/go/bin/trufflehog}"
[ -x "$TRUFFLEHOG" ] || TRUFFLEHOG="$(command -v trufflehog || true)"
if [ -n "$TRUFFLEHOG" ] && [ -x "$TRUFFLEHOG" ]; then
  "$TRUFFLEHOG" git "file://$MAIN_WT" --since-commit="$BASE" --branch="$BRANCH" --fail \
    && echo "trufflehog: clean over $RANGE" \
    || { echo "ABORT: trufflehog flagged secrets in $RANGE"; exit 1; }
else
  echo "trufflehog not installed; skipping. Rely on the grep heuristic above."
fi
```

##### Asset names & personal paths — scan commit *messages*, not just the diff

Private model/dataset names and personal paths leak through **commit messages** and **source
comments** as easily as through code. Public history should describe *what* changed generically
("the 805 GB DeepSeek-V4 checkpoint", "a per-rank run log"), never a private path under
`~/models` / `~/data` or a RIKEN allocation path.

```bash
# (a) Personal / allocation paths in messages AND source — must be empty.
{ git log "$RANGE" --format='%B'; \
  git diff "$RANGE" -- 'a64fx/*' 'ryzen/*' 'zen2/*' 'cuda/*' 'vulkan/*' 'common/*' 'tools/*'; } \
  | grep -inE '/home/[a-z]|/Users/[A-Za-z]|C:\\Users\\|/vol0006/|hp250467' \
  && echo "ABORT: personal/allocation path in messages or source" \
  || echo "clean: no personal paths"

# (b) Model/asset filename tokens in messages + added source — EYEBALL each; strip any that
#     names a private checkpoint/dataset. Generic names / public samples are fine.
{ git log "$RANGE" --format='%B'; \
  git diff "$RANGE" -- 'a64fx/*' 'cuda/*' 'common/*' 'tools/*' | grep '^+'; } \
  | grep -oiE '[A-Za-z0-9_.-]{3,}\.(gguf|safetensors|bin|pt|pth|npy|usdz|glb|gltf)' \
  | sort -u
```

Any private name → rewrite before pushing (`git filter-repo --replace-message`, or amend/rebase
for source comments). **Re-run every Check 1 scan after a rewrite — the commit hashes change.**

#### Check 2 — No build artifacts

Build outputs explode pack size, conflict on every rebuild, and embed absolute machine paths.
Common culprits here:

- `build/`, `a64fx/llm/build/`, `vulkan/build/`, `*/build_*/` — make / cmake / ninja output trees.
- `*.a`, `*.o`, `*.obj`, `*.so`, `*.dylib`, `*.dll`, `*.lib`, `*.pdb` — compiled objects / libraries.
- `*.ninja_deps`, `*.ninja_log`, `build.ninja`, `CMakeCache.txt`, `CMakeFiles/`,
  `CTestTestfile.cmake`, `compile_commands.json` — cmake/ninja state (vulkan).
- `__pycache__/`, `*.pyc`, `*.egg-info/`, `cuda/trellis2/model_root/`.
- Compiled test/bench binaries (`a64fx/llm/ds4f_*_test`, `bench`, `test_*` with no extension) and
  per-rank run output (`a64fx/llm/ds4f_ep_*_rank*.txt`, `output.*/`, `*.log`).

```bash
git diff --name-only "$RANGE" \
  | grep -E '(^|/)(build|build_[a-z0-9]*)/|/CMakeFiles/|\.(a|o|obj|so|dylib|dll|lib|pdb|pyc|ninja_deps|ninja_log)$|/CMakeCache\.txt$|/CTestTestfile\.cmake$|/build\.ninja$|/compile_commands\.json$|(^|/)model_root/' \
  && { echo "ABORT: build artifacts staged for push"; exit 1; } \
  || echo "No obvious build artifacts in $RANGE"
```

`.gitignore` already covers these; a hit usually means a `git add -f` or a mistake rebased
forward. Drop it; do not push it.

#### Check 3 — No unintended binary / asset data

Model weights and captured tensors live in `~/models` / `~/data` (external) or external storage —
never in repo history. Reject:

- Model weights: `*.gguf`, `*.safetensors`, `*.bin`, `*.pt`, `*.pth` (any non-trivial size).
- Reference dumps: `*.npy` / `*.npz` (e.g. `rdna4/trellis2/*.npy`, `ref/*/dumps/`), large captures.
- Images over ~256 KB: `*.png`, `*.jpg`, `*.jpeg`, `*.exr`, `*.hdr`, `*.tif` (small spec/icon
  fixtures are fine).
- Extracted kernel blobs (`rdna4/fp8/*.co`), `.codex` / `.claude/` scratch dirs, `third_party/`.
- Anything you cannot trivially regenerate from source.

```bash
git diff --stat "$RANGE" | awk '$3 ~ /^[0-9]+$/ && $3+0 > 256 { print }' | head
# Inspect anything large; binaries show as "Bin <n> -> <m> bytes".

git diff --name-only "$RANGE" \
  | grep -iE '\.(gguf|safetensors|bin|pt|pth|npy|npz|blend|fbx|glb|gltf|abc|exr|hdr|tif|tiff|co)$|(^|/)dumps/|(^|/)\.(codex|claude)/' \
  && { echo "ABORT: unexpected binary/asset paths in $RANGE"; exit 1; } \
  || echo "No flagged binary/asset paths in $RANGE"
```

If a binary must ship, the answer is external storage (`~/data`, an asset bucket, git-lfs) — never
a plain `git add`. When in doubt, ask.

#### Check 4 — User permission (always, no exceptions)

After the three audits pass, **stop and ask before pushing** — even if the user earlier said
"push when done" or approved many pushes today (the single-shot rule at the top of this section).
The order is always: audit → summarize → ask → push. When asking, summarize:

- branch and remote (`origin/ds4p` on `github.com:lighttransport/gemm`),
- fast-forward vs force-push (force: also list which previously-public commit hashes get orphaned),
- commit count + a one-line per-commit summary,
- audit results from Checks 1–3 ("no credential strings, no build artifacts, no flagged binaries"),
- any pre-push test/benchmark results worth surfacing.

Prefer `--force-with-lease` over `--force` (lease refuses if the remote moved since your last
fetch). The push itself is its own separate `ssh fugaku '... git push ...'` invocation — never
chained after a commit.

### After the push

- Force-push: immediately note the previous remote tip in the conversation so a teammate can
  recover (`git fetch origin <old-sha>:refs/heads/recovered-<branch>` on their side).
- Open / update the PR if one exists; the default PR base is `main`.
- Do not delete branches the user did not explicitly ask to delete.

### If a check fails

- Credential leak — **stop**, surface it, and treat the credential as compromised even if the
  commit is still local. Rotate first, scrub history second; never push to "clean it up later"
  (we already moved `DS4P_BASH_HTTP_TOKEN` to the env for exactly this reason).
- Stray binary/artifact — drop it from history (`git filter-repo --invert-paths --path <p>`, or
  rebase for a single recent commit), update `.gitignore`, then re-run the full checklist.
- Unintended asset — confirm with the user whether it belongs in external storage or should be
  removed.

The pre-push checklist is non-negotiable: one large binary in the remote permanently bloats every
future clone, and one leaked key is a security incident regardless of how fast it gets rotated.
