# GenIA reference tools

This directory pins the upstream source and provides **math-level** oracles.
It does not yet capture a complete GenIA reconstruction or decoder/loss backward.
Production vhuman code imports none of these tools or upstream frameworks.

Fetch the source into repository-local scratch (do not use `/tmp`):

```sh
git clone --branch release https://github.com/facebookresearch/GenIA.git tmp/genia-upstream
git -C tmp/genia-upstream checkout 6fbd691ac90248eedb16e1099f2c7bef6831c69c
```

`dump_math.py` authenticates the checked-out commit and visibility-source hash.
It executes the pinned upstream DDA function bodies with only JIT decorators
removed; NumPy supplies independent fusion and guidance expected values. No
Torch, Numba, checkpoint or GPU is needed to produce these fixtures.

```sh
python3 ref/genia/dump_math.py
make -C cuda/genia all
XDG_CACHE_HOME="$PWD/tmp/genia-cache" TMPDIR="$PWD/tmp/genia-runtime" \
  python3 ref/genia/verify_math.py
```

The verifier checks lock identity, tensor hashes, shapes and dtypes before
comparison. It writes a JSON receipt only on successful comparison. Fixtures
and receipts default to `tmp/genia-ref/math/` and are not committed.

Export the missing SLAT conditioning fuser using an existing offline Torch
interpreter (the tool writes F32 safetensors without an extra serializer dependency):

```sh
tmp/vhuman-musclemimic-py312/bin/python ref/genia/export_conditioning.py \
  --checkpoints /mnt/nvme02/models/sam3d/checkpoints
```

Output defaults to `MODELS_ROOT/genia/native/` (`/mnt/nvme02/models/genia/native/` here); an existing export is not overwritten.
Its manifest records source/output hashes, dimensions and comparisons with SS.
All 11 shared tensors in the local SS/SLAT checkpoints differ, so substituting
the already-converted SS fuser would produce the wrong conditioning.

The upstream release's full reference recipe requires its patched SAM3D
submodule, PyTorch 2.7.1/cu128 and CUDA extensions. Those dependency requirements
are recorded in `lock.json`; they are not installed by the math tools. Existing
checkpoints under `/mnt/nvme01/models/sam3d/checkpoints/` should be reused for
future scene captures. Missing variants must be exported from those originals,
not substituted silently. No model downloader runs automatically.

Upstream source is [CC BY-NC 4.0](https://github.com/facebookresearch/GenIA/blob/release/LICENSE).
SAM3D, gsplat, nvdiffrast and other components retain their own licenses;
see [the upstream notices](https://github.com/facebookresearch/GenIA/blob/release/THIRD_PARTY_LICENSES.md).
