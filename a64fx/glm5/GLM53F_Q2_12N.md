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
