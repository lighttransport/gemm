# RDNA4 Qwen3.8 regression gates

These gates separate deterministic long-context performance from serving
quality. Run them after changes to Q8 KV storage, prefill, decode graphs,
DFlash2 verification, snapshots, or cache injection.

## Randomized 64K Q8/Q8 gate

Build `test_hip_llm`, then run the ordinary target:

```sh
cd rdna4/llm
QWEN38_MODEL=/path/to/target.gguf \
  QWEN38_LONG_MODE=target ./test_qwen38_long_context.sh
```

Run the sidecar configuration independently, or set the mode to `both`:

```sh
QWEN38_MODEL=/path/to/target.gguf \
QWEN38_DFLASH2_MODEL=/path/to/dflash2.gguf \
  QWEN38_LONG_MODE=dflash2 ./test_qwen38_long_context.sh
```

The gate fills a 65,536-token prefix with deterministic random tokens, uses
Q8 for both K and V, repeats the measured suffix in one process, and requires
one stable suffix hash. Defaults require at least 400 tok/s prefill and 32
tok/s decode. Override `QWEN38_LONG_PREFILL_FLOOR` or
`QWEN38_LONG_DECODE_FLOOR` only when recording a deliberately different
hardware baseline. Detailed logs remain under `tmp/long-context-gate/`.

On the RX 9070 XT on 2026-09-22, two 128-token suffix repeats measured:

| Mode | Prefill | Minimum decode | Suffix hash |
|---|---:|---:|---|
| target | 447.15 tok/s | 35.50 tok/s | `aed3c962c4a6525d` |
| DFlash2 configured | 444.69 tok/s | 35.54 tok/s | `0e3dfc3705bbe26b` |

Both used random-prefix hash `90178de69a24a76e`. DFlash2 deliberately
reported target-only decode after position 65,537. Its suffix differs from a
target-only process, so these results establish repeatability and performance,
not cross-mode byte parity.

## Resident serving quality

Run the target plus DFlash2 gate with enough context for coding responses:

```sh
python3 test_qwen35_dflash2_http.py \
  --model /path/to/target.gguf \
  --sidecar /path/to/dflash2.gguf \
  --runner ./test_hip_llm --context 1024 --dflash2-draft 7
```

The suite covers direct stdio and HTTP, greedy and seeded sampling, repeated
cache hits, A/B/A restoration, LRU eviction, disconnect and explicit
cancellation, recovery, concurrent cache identities, and multi-turn coding.
Generated C++ is compiled and run. Algorithmic cases cover lower-bound
indices, inclusive clamping, and stable string deduplication.

Run the CPU-only contracts before a resident test:

```sh
python3 test_codex_protocol.py
python3 test_dflash_overlap_lifecycle.py
bash test_qwen38_profiles.sh
```

## Experiment promotion rules

Production defaults are the paths selected without diagnostic environment
variables. Experimental fusion, projection, attention, selector, and
cache-injection overlap switches remain opt-in until they pass all applicable
gates above with identical required hashes and a repeatable throughput gain.

Cache-injection overlap has per-stream scratch, fail-closed setup, abort
retirement, reset/restore fences, and idempotent teardown coverage. It remains
opt-in because its measured throughput has been neutral or slightly slower.
Candidates that change reduction order remain diagnostics even when a short
quality prompt happens to pass.
