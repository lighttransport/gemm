# GLM-5.3-Flash Q2 routed decode on 12 A64FX nodes

This path consumes the mixed-IQ routed experts from
`~/models/glm53f-gguf/GLM-5.3-Flash-UD-Q2_K_XL-00001-of-00004.gguf` directly.
The compact attention/dense core and shared expert still come from the existing
GLM-5.3 safetensors-derived rank images.  It is therefore a Q2-routed hybrid,
not yet a fully GGUF-backed graph.

Run it inside an allocated 12-node interactive job:

```sh
sh a64fx/glm5/run_glm53f_q2_12n.sh
```

The launcher fixes the Fujitsu MPI environment, builds with `mpifcc -Nclang`,
stages all large files under node-local `/local`, creates the uTofu topology,
and enforces a 20 tok/s default gate.  Completed Q2 images are reused when the
manifest layer range and blob size match.

## Layout and memory

GGML IQ blocks are 256 columns wide.  Each routed expert is consequently split
into eight 256-wide parts rather than twelve partial blocks.  Part `p` of
expert `e` is owned by rank `(e + offset[p]) % 12`, with offsets
`0,1,3,4,6,7,9,10`.  This balances 2,304 expert parts evenly: 192 per rank and
layer.

The full layers 3 through 44 image is 8,252,817,408 bytes per rank.  Staging
uses bounded positional I/O, periodically syncs output, advises source and
destination pages away, and atomically renames the completed blob/manifest.
During the validated run rank 0 retained about 19.5 GiB `MemAvailable` after
loading the 7.686 GiB routed image and 0.062 GiB shared image.

## Results (job 51321372, 2026-09-04)

All figures use 128 single-stream decode steps, 12 ranks, the hardware OpenMP
barrier, uTofu FP32 all-reduce, and finite-logit/PASS validation.

| Configuration | tok/s | ms/token | Final token |
| --- | ---: | ---: | ---: |
| Conservative math, 47 threads | 17.629--17.749 | 56.34--56.73 | 279 |
| Fast math, 47 threads | 20.612--20.854 | 47.95--48.52 | 4362 |
| Fast math, 48 threads, best | 21.757 | 45.96 | 4362 |
| Fast math, 48 threads, repeat | 19.154--19.918 | 50.21--52.21 | 4362 |
| Fast math, 47 threads, 2-D uTofu | 20.845 | 47.97 | 7271 |
| Fast math, 47 threads, BF16 all-reduce | 20.611 | 48.52 | 16120 |

The 48-thread and alternate-collective results are too variable or alter the
reduction trajectory without a repeatable speedup, so production defaults stay
at 47 threads and one-dimensional FP32 uTofu.  Fast math is the default needed
to clear 20 tok/s, but it diverges from conservative greedy decode at step 1;
set `GLM53F_FAST_MATH=0` when conservative numerical behavior is required.

The fast 47-thread profile measured 6.61 ms/token MHC, 25.52 attention, 15.16
FFN, and 1.47 head.  Further repeatable gains should target attention rather
than the mixed-IQ routed kernel; the isolated Q2 routed layer measured
0.254--0.269 ms/token including its collective.

## 30 tok/s investigation

The following full 128-step experiments were rejected on the same job:

| Variant | tok/s | Finding |
| --- | ---: | --- |
| KDA output projection, 8 rows/dot | 20.089 | Attention regressed to 26.23 ms/token. |
| KDA output projection, 4 rows/dot | 20.353 | Attention regressed to 26.09 ms/token. |
| `-mcpu=a64fx` | 20.886 | Accepted and exact trajectory, but within allocation variance. |

Speculative MTP using the staged safetensors layer 45 against the Q2 target
accepted 31/32 drafts (96.875%), but did not raise throughput: M=2 target
verification took 103.78 ms/cycle, or 51.91 ms per position, yielding only
17.623 delivered tok/s.  The current Q2 batch MoE falls back to one
single-token IQ evaluation per position because independently routed experts
rarely overlap.  Reaching 30 tok/s requires a structural attention reduction
(for example channel-level KDA partitioning) or a genuine multi-token IQ MoE
verifier; compiler flags and output-row batching have been exhausted.

## Long-context sparse decode (job 51351748, 2026-09-05)

The sparse MLA path now follows the incremental pooling strategy from
llama.cpp PR 27754: each completed four-token index key is compressed once,
instead of recompressing the complete prefix on every decode step. Above the
2,048-token selection budget, pool scoring is divided across 12 ranks and
exchanged with the FP32 uTofu all-reduce. An exact bounded heap retains the top
512 pools, and selected latent rows are packed contiguously before MLA.

Each rank owns only five or six MLA heads. The long-context kernel now uses
eight contiguous token shards per local head inside one OpenMP team, then
reduces the partial value vectors in a fixed order. A 128-token greedy run
after an 8,378-token prompt produced stable windows of **20.199** and
**20.251 tok/s**, **20.226 tok/s** overall decode, and the same 128 token IDs
as the full-sort control. Decode-only phase averages were 6.020 ms MHC,
28.199 ms attention, 14.670 ms FFN, and 1.210 ms head.

Generation IDs are buffered and written once after timing; rank 0 previously
reopened the shared-filesystem output for every token. Profiling now resets at
the prompt/decode boundary. A longer run consumed the same 8,378-token prompt,
generated 7,272 tokens to EOS, and averaged 19.256 tok/s while total context
grew to 15,650 tokens.

The semantic gate used `prompts/glm53f_cpp_codegen_task.md`. A coherent raw
C++17 generation reached EOS but strict compilation found one unused helper.
Deleting only that helper made it compile with
`mpiFCC -Nclang -std=c++17 -O2 -Wall -Wextra -Werror -pedantic`; the ARM binary
then passed FIFO, duplicate, cancellation, empty-payload, integer-range,
malformed-command, and quit tests with an exact output diff.
