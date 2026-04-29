# MHR Skinning Port — Research & Plan (Step 6 blocker)

Goal: produce `common/sam3d_body_mhr.h` (pure C) so the CPU runner can turn
the regressed 519-d pose vector into `pred_vertices (B,18439,3)` and
`pred_keypoints_3d (B,70,3)`. This document scopes that work — it is a
decision framework + kickoff punch list, not a full design.

Reference: `/tmp/sam-3d-body/sam_3d_body/models/heads/mhr_head.py:271`
(`MHRHead.forward` → `mhr_forward` → `self.mhr(...)`).
The `self.mhr` ScriptModule lives at
`/mnt/disk01/models/sam3d-body/dinov3/assets/mhr_model.pt`. Every scripted
helper its `forward` calls is embedded as Python source inside that .pt
zip and was extracted to `/tmp/mhr_jit_code/__torch__/pymomentum/` for
this analysis.

---

## 1. What does `mhr_model.pt` actually do?

It's a `pymomentum.mhr.MHR` ScriptModule (Meta's "Momentum" rig runtime,
torch backend). Top-level `forward`:

```
forward(identity_coeffs:(B,45),     # = "shape" PCA coeffs (45)
        model_parameters:(B,204),   # = mhr_model_params from mhr_head.py
        face_expr_coeffs:(B,72),    # = "face" PCA coeffs (72)
        apply_correctives:bool=True
       ) -> (verts:(B,18439,3), skel_state:(B,127,8))
```

Skel-state convention: per-joint 8-vec `(tx,ty,tz, qx,qy,qz,qw, s)`
(translation + unit quaternion + uniform scale).

### Sub-module call graph (extracted from the jit zip)

```
mhr.forward
├── character_torch.blend_shape.forward(shape)         identity rest pose: einsum 'nvd,...n -> ...vd' over (45,18439,3) + base_shape(18439,3) -> (B,18439,3)
├── character_torch.parameter_transform.forward(cat(model_params(204), zeros(45)))
│        einsum 'dn,...n->...d' with (889,249) sparse-ish dense matrix -> (B,889) joint_params
├── character_torch.skeleton.joint_parameters_to_local_skeleton_state(joint_params)
│        reshape -> (B,127,7); per-joint:
│          local_t = jp[..,:3] + joint_translation_offsets       (127,3)
│          local_q = euler_xyz_to_quaternion(jp[..,3:6])         (127,4) (w-last)
│          local_q = qmul_normalized(joint_prerotations, local_q)
│          local_s = exp(jp[..,6] * ln(2))   # i.e. 2^jp[6], scalar scale
│        cat -> (B,127,8)
├── character_torch.skeleton.local_skeleton_state_to_skeleton_state(local_state)
│        Walk skeleton tree as 4 prefix-product stages
│        (skeleton._pmi_buffer_sizes = [65,56,62,83], total 266 = 2*133),
│        pmi tensor of shape (2,266) packs (target_idx, source_idx) pairs.
│        For each pair: state[source] = skel_multiply(state[target], state[source]).
│        skel_multiply((t1,q1,s1),(t2,q2,s2)) =
│            t = t1 + s1 * rotate_by_quat(q1, t2)
│            q = qmul(q1, q2)
│            s = s1 * s2
│        Done in fp64 internally then cast back -> global_skel_state (B,127,8)
├── face_expressions_model.forward(face)
│        einsum 'nvd,...n->...vd' with (72,18439,3) -> (B,18439,3) (no base)
├── pose_correctives_model.forward(joint_params)         (only if apply_correctives)
│        feat = batch6DFromXYZ(joint_params[:,2:127, 3:6])     # (B,125,6) — XYZ-Euler → 6D rotation columns
│        feat[:,:,0] -= 1; feat[:,:,4] -= 1    # subtract identity so zero pose => zero feat
│        feat = flatten -> (B, 750)
│        h = SparseLinear(feat) ; weight is COO (3000,750), 53136 nnz
│        h = ReLU(h)
│        out = Linear(3000, 55317)(h)         # 55317 = 18439*3
│        reshape -> (B,18439,3)
├── linear_model_unposed = identity_rest_pose + face_expr (+ pose_correctives)
└── character_torch.skin_points(skel_state, linear_model_unposed)
        joint_state = skel_multiply(global_skel_state, inverse_bind_pose)        (B,127,8)
        for k in range(51337):
            v = vert_indices_flattened[k]; j = skin_indices_flattened[k]; w = skin_weights_flattened[k]
            skinned[v] += w * skel_transform_points(joint_state[j], rest_vert[v])
        # skel_transform_points((t,q,s), p) = t + rotate_by_quat(q, s*p)
        -> verts (B,18439,3)
```

### Op set (jit_code references)

* `__torch__.pymomentum.quaternion`: `euler_xyz_to_quaternion`,
  `multiply_assume_normalized`, `inverse`, `conjugate`, `normalize`,
  `rotate_vector{,_assume_normalized}`. Quaternion convention is **(x,y,z,w)
  with w last**.
* `__torch__.pymomentum.skel_state`: `multiply`, `transform_points`,
  `inverse`, `_normalize_split_skel_state`, `_multiply_split_skel_states`.
* `__torch__.pymomentum.backend.skel_state_backend`:
  `global_skel_state_from_local_skel_state` (the prefix-product walker)
  and `skin_points_from_skel_state` (LBS).
* `__torch__.pymomentum.mhr.utils.batch6DFromXYZ` — XYZ-Euler → 6D
  (concat of first two columns of XYZ rotation matrix).

### What `mhr_head.mhr_forward` passes in (mhr_head.py lines 163–229)

```
shape_params  = pred_shape (B,45)         -> identity_coeffs
face_params   = pred_face  (B,72) * 0     -> face_expr_coeffs (always zero in inference)
model_params  = cat(full_pose_params (B,136), scales (B,68))  (B,204) -> model_parameters
  full_pose_params = cat(global_trans*10 (B,3), global_rot_euler (B,3), body_pose_euler[:,:130]) -> B,136
       with hand-PCA decoded (replace_hands_in_pose) into joints hand_joint_idxs_{left,right}
  scales           = scale_mean(68) + scale_params(B,28) @ scale_comps(28,68)
```

Then `verts /= 100; joint_coords /= 100; verts[...,1:3] *= -1; j3d[...,1:3] *= -1`
(meters + camera-axis flip), and finally `j3d = (keypoint_mapping (308,18566)
@ cat(verts, joint_coords))[:,:70]`. The 308→70 slice is hardcoded in the
head; the keypoint_mapping is **already in our safetensors** as
`sam3d_body_mhr_head.safetensors::keypoint_mapping` per PORT.md.

### Key constants we will need to dump from mhr_model.pt

These are **not** in the existing safetensors and must be exported once:

| Tensor                                       | Shape         | dtype   | Source path in jit                                          |
|----------------------------------------------|---------------|---------|-------------------------------------------------------------|
| `blend_shape.shape_vectors`                  | (45,18439,3)  | f32     | `character_torch.blend_shape.shape_vectors`                |
| `blend_shape.base_shape`                     | (18439,3)     | f32     | `character_torch.blend_shape.base_shape`                   |
| `face_expressions.shape_vectors`             | (72,18439,3)  | f32     | `face_expressions_model.shape_vectors`                     |
| `parameter_transform`                        | (889,249)     | f32     | `character_torch.parameter_transform.parameter_transform`   |
| `joint_translation_offsets`                  | (127,3)       | f32     | `character_torch.skeleton.joint_translation_offsets`       |
| `joint_prerotations` (quaternions xyzw)      | (127,4)       | f32     | `character_torch.skeleton.joint_prerotations`              |
| `pmi`                                        | (2,266)       | i64     | `character_torch.skeleton.pmi`                             |
| `pmi_buffer_sizes` (split lengths)           | [65,56,62,83] | scalars | `character_torch.skeleton._pmi_buffer_sizes`               |
| `inverse_bind_pose` (skel_state per joint)   | (127,8)       | f32     | `character_torch.linear_blend_skinning.inverse_bind_pose`  |
| `skin_indices_flattened`                     | (51337,)      | i32     | `character_torch.linear_blend_skinning.skin_indices_flattened` |
| `skin_weights_flattened`                     | (51337,)      | f32     | `…linear_blend_skinning.skin_weights_flattened`            |
| `vert_indices_flattened`                     | (51337,)      | i64     | `…linear_blend_skinning.vert_indices_flattened`            |
| `pose_correctives.sparse_indices`            | (2,53136)     | i64     | `pose_correctives_model.pose_dirs_predictor.0.sparse_indices` |
| `pose_correctives.sparse_weight`             | (53136,)      | f32     | `…0.sparse_weight`                                          |
| `pose_correctives.sparse_shape`              | (3000,750)    | scalars | hardcoded in jit code                                       |
| `pose_correctives.linear_weight`             | (55317,3000)  | f32     | `…2.weight`                                                 |
| `pose_correctives.linear_bias`               | (55317,)      | f32     | `…2.bias` (assumed present; verify on extract)              |

Total weight footprint, f32: 45·18439·3 + 72·18439·3 + 18439·3 + 889·249
+ 127·3 + 127·4 + 127·8 + 51337·(4+4) + 53136·8 + 53136·4 + 55317·3000
+ 55317  ≈  **~915 MB** dominated by the 663 MB Linear(3000,55317).
Storing the Linear in fp16 cuts it to **~415 MB total**; fp16 mat-vec
should be safe given the f32 epsilon target is ~1e-3 mm scale.

---

## 2. Minimal op set for a C reimpl

### (a) Trivial linear algebra (already in `cpu_compute.h`)

* `matvec` for the (889,249) parameter_transform — but only ~368 nnz:
  loop is 249·889 = ~221k MACs/frame, or 368 if exploiting sparsity.
* `matmul` for the dense Linear(3000→55317) — **the heavy hitter**, ~166M
  MACs. fp32 dgemv-style with our existing `cpu_compute_matmul` is fine.
* `index_select` / `index_add` / scatter-add for skinning (51337 entries).
* `einsum 'nvd,...n->...vd'` over shape vectors — equivalent to
  `gemv: shape_vectors.reshape(N, 18439*3) ^T @ coeffs(N,)` → 18439·3
  output. Two such (45-comp identity + 72-comp face) ≈ 6.5M MACs.
* `einsum 'da,ab->db'` for hand PCA (54×54) — trivial.

### (b) SMPL-X-like geometry kernels (need writing once, share with future ports)

* `quat_from_euler_xyz` (intrinsic XYZ → unit quaternion, w-last).
* `quat_mul` (Hamilton product, w-last).
* `quat_rotate_vec` (`v + 2 * (axis × (axis × v + r·v))`).
* `skel_state_multiply` (compose two (t,q,s) 8-vecs).
* `skel_state_transform_points` (`t + rotate(q, s·p)`).
* Skeleton **prefix-product walker**: 4 stages of (source,target) pairs
  from `pmi`; replaces standard recursive forward kinematics. Crucial:
  must run in fp64 internally to match jit (verify this is necessary —
  may relax to fp32 if epsilon allows).
* LBS: scatter-add `skinned[v] += w * skel_state_transform_points(joint_state[j], rest[v])`.

### (c) Momentum-specific that we'd hand-roll

* `model_parameters_to_joint_parameters` is just the (889,249)
  matvec — **not custom**, it's the parameter_transform table. The "rig"
  knowledge is baked into that matrix. Good news: zero custom code.
* `pose_correctives` SparseLinear: COO sparse-matvec
  `out[i] = Σ sparse_weight[k] * x[ idx_col[k] ]` with
  `idx_row[k]==i`. ~53k MACs/frame, trivial.
* `pose_correctives._pose_features_from_joint_params`: per-joint slicing
  of the joint_parameter (B,127,7) tensor (skip joints 0–1, take euler
  3:6 of joints 2..127), `batch6DFromXYZ`, subtract identity. All scalar
  ops on (125,3) → (125,6).

There is **no** dependency on `roma`, `pymomentum.mhr.MHR`, libtorch ops,
or any momentum-specific binary kernel. The .pt zip ships pure-Python
scripted ops; no `torch.ops.pymomentum.*` C++ extensions are invoked at
runtime.

### Outside `mhr_model.pt` but inside `mhr_head.py`

* `rot6d_to_rotmat` (Gram-Schmidt; mhr_head.py imports it).
* `rotmat_to_euler("ZYX", R)`: from `roma`. Easy: extract `roll = atan2(-R[1,2], R[1,1])` etc. (will write 10 lines of trig).
* `compact_cont_to_model_params_body` / `compact_cont_to_model_params_hand`
  — index gather + `batchXYZfrom6D` (atan2-from-rotmat) + `atan2(sin,cos)`
  for 1-DoF axes. Index tables are constants in `mhr_utils.py`; copy them.
* `replace_hands_in_pose`: hand-PCA decode + scatter into 27-idx slots
  on the (B,136) `full_pose_params` vector (twice — left, right).
* `mhr_param_hand_mask` zero-out (133-bit mask, hardcoded).
* `keypoint_mapping @ cat(verts, joints)` final regression: a (308,18566)
  by (18566,3) matmul; slice first 70 keypoints; flip Y/Z.

---

## 3. Port strategy

| Option                                 | Verdict | Effort | Pros | Cons |
|----------------------------------------|---------|--------|------|------|
| (i) Pure C reimpl from jit code        | **PICK** | ~2 d   | Zero new deps; matches codebase rule (pure C / NVRTC); future CUDA path is a straight kernel port; correctness reachable because jit is fully visible. | One-shot extraction of ~915 MB of weights to safetensors; we own the geometry kernels. |
| (ii) Extract weights + hand-code SMPL-X-like forward | (=i)   | (=i)   | Same as (i). | Conceptually identical to (i); listing it separately just because the pipeline is "SMPL-X-shaped" but the rig is Momentum, not SMPL-X — naming-only distinction. |
| (iii) Link libtorch C++ for this module| Reject  | ~1 d to wire, +days for build/CI | Drop-in `torch::jit::load`; no port effort. | **Hard violation of project rule** ("pure C / NVRTC, no libtorch"). Adds ~600 MB toolchain, blows up our static-link story, complicates A64FX cross-compile, no clean CUDA reuse path. |
| (iv) ONNX export + onnxruntime         | Reject  | unknown | If exportable, no port; ORT is C-callable. | Sparse-COO + scatter ops are flaky-to-impossible to ONNX-export from a torch.jit module; would still bring a 30 MB ORT dep; same project-rule violation; CUDA story competes with our NVRTC plan. |

**Recommendation: option (i).** The jit module is small enough (≈12
distinct ops, all in fp32) that a single-header `common/sam3d_body_mhr.h`
mirroring the jit's call graph is the cleanest fit. The weight dump is
mechanical (≈40 lines of Python, mirror `convert_ckpt.py`).

---

## 4. Smallest first-cut unit test (verify_mhr.c bring-up plan)

We have all reference dumps at `/tmp/sam3d_body_ref/mhr_params__*.npy`.
Verify in a stack: each stage's output is also dumpable from a sidecar
Python script, so we add intermediate dumps (`mhr_step_*.npy`) before
writing C.

Suggested staircase of `verify_mhr_stage{N}.c` (smallest reliable unit
first):

| Stage | C kernel                                  | Inputs (npy)                                              | Outputs (npy)                          | Tol     | Effort  |
|-------|-------------------------------------------|------------------------------------------------------------|----------------------------------------|---------|---------|
| 1     | `rot6d_to_rotmat` (3 lines, Gram-Schmidt) | `pred_pose_raw[:6]`                                        | `global_rot_rotmat`                    | 1e-6    | 0.5 h   |
| 2     | `rotmat_to_euler_ZYX`                     | rotmat                                                     | `global_rot`                           | 1e-5    | 0.5 h   |
| 3     | `compact_cont_to_model_params_body` (260→133)| `pred_pose_raw[6:]` (260)                                | body_pose_euler (1,133)                 | 1e-5    | 1 h     |
| 4     | `compact_cont_to_model_params_hand` ×2 + `replace_hands_in_pose` | hand 108                              | full_pose_params (1,136)                | 1e-5    | 1 h     |
| 5     | scale PCA `mean(68)+scale@scale_comps` and assemble (1,204) | scale (1,28)                          | `mhr_model_params` (1,204)              | 1e-5    | 0.3 h   |
| 6     | `parameter_transform` matvec               | model_params(1,249)                                       | joint_parameters (1,889)                | 1e-5    | 0.3 h   |
| 7     | local skel state (translation+quat+scale)  | joint_parameters                                          | local_skel_state (1,127,8)              | 1e-5    | 1 h     |
| 8     | prefix-product walker (4 stages, fp64 acc) | local_skel_state, pmi, _pmi_buffer_sizes                  | global_skel_state (1,127,8)             | 1e-4    | 2 h     |
| 9     | quat→rotmat × `[2,1,0]` permutation for `joint_global_rots` | global_skel_state                | `joint_global_rots` (1,127,3,3)         | 1e-5    | 0.3 h   |
| 10    | shape blend (45) + face blend (72) + pose correctives (sparse 750→3000→ReLU→Linear→55317) | shape, face, joint_params      | linear_model_unposed (1,18439,3)        | 1e-4    | 2 h     |
| 11    | LBS skin (51337 entries scatter-add)      | global_skel_state, linear_model_unposed                   | `pred_vertices` (1,18439,3)             | 1e-3 mm | 1.5 h   |
| 12    | keypoint regression (308,18566)@cat(verts,joints), slice :70, axis flip | verts, joint_coords            | `pred_keypoints_3d` (1,70,3)            | 1e-4    | 0.5 h   |

**Smallest unit that is non-trivial** = stage 7 (local skel state from joint
parameters). It exercises euler→quat, quaternion multiply, log-scale, and
dies if any quaternion convention (w-first vs w-last) is wrong. It's a
single 127-iteration loop with no reductions across joints.

**Smallest end-to-end checkable thing** = run stages 1–7 only with
`apply_correctives=False, face=zeros, identity=zeros`; LBS will produce a
T-pose-with-no-shape mesh that we can sanity-check against `rest_vertices`
+ skinning, before pulling in the 663 MB Linear of stage 10.

**Recommended first PR scope:** stages 1–9 + a "no-correctives" path of
stage 10 (skip pose_correctives) + stage 11 + stage 12 — that gets
verts/keypoints close to ref except for ≈mm-scale corrective deltas.
Then add pose_correctives in a second PR.

---

## 5. Effort estimate (person-hours)

| Option                                    | One-off setup | Core port  | Verify+debug | Polish  | **Total** |
|-------------------------------------------|---------------|------------|--------------|---------|-----------|
| (i) Pure C reimpl (recommended)           | 2  (asset dump + safetensors emit) | 7 (stages 1–12) | 4 (epsilon chase, fp64 prefix-product) | 1 | **~14 h** |
| (ii) Same as (i) — naming variant         | 2             | 7          | 4            | 1       | **~14 h** |
| (iii) libtorch C++ link                   | 6 (CMake + cross-compile A64FX libtorch — risky) | 1 (10-line wrapper) | 1 | 0 | **~8 h, but blocked on rule violation** |
| (iv) ONNX + onnxruntime                   | 4 (try export — likely fails on sparse + scatter) | unknown | unknown | unknown | **~8 h to find out it doesn't work, then fall back to (i)** |

Estimates assume the reference dumps already cover the intermediate
states we need; if they don't, allow +2 h to extend the Python sidecar
to dump `pmi`-walked global state, joint_state per joint, and the
pose_correctives intermediate (B,3000) feature.

---

## Kickoff punch list

1. Add `dump_mhr_assets.py` (sibling of existing `convert_ckpt.py`):
   `torch.jit.load(mhr_model.pt)` → walk `state_dict()` and the few
   non-Parameter buffers (`pmi`, `_pmi_buffer_sizes`, `sparse_shape`),
   emit a single `sam3d_body_mhr_jit.safetensors` and a small JSON for
   the scalar shape constants. Also fp16-pack the (55317,3000) Linear if
   the file size is a concern (already 415 MB).
2. Extend the Python ref dumper to write per-stage `mhr_stage{N}.npy`
   files matching the table above (use the input `mhr_params__*.npy`
   we already have as ground truth and run the jit module piece by piece
   with hooks on its sub-modules).
3. New header `common/sam3d_body_mhr.h` containing:
   * `s3db_mhr_assets` struct (pointers to all extracted weights).
   * `s3db_mhr_load(path) / free`.
   * `s3db_mhr_pose_raw_to_model_params(pred_pose_raw[266], scale[28], hand[108], face[72], out_model_params[204])` — stages 1–5.
   * `s3db_mhr_forward(model_params[204], identity[45], face[72], apply_correctives, out_verts[18439*3], out_skel_state[127*8])` — stages 6–11.
   * `s3db_mhr_keypoints(verts, skel_state, keypoint_mapping, out_kpts[70*3])` — stage 12.
4. New `verify_mhr_stage{N}.c` per row (mostly clones of existing
   `verify_mhr_head.c`).
5. Wire the new calls into `sam3d_body_run_mhr` in
   `sam3d_body_runner.{c,h}` (already stubbed).
6. CUDA port: each kernel is independently shape-uniform — quat ops and
   prefix-product walker map cleanly to NVRTC; the (3000→55317) GEMV is
   cuBLAS-shaped; LBS scatter-add wants atomicAdd. Defer until CPU is
   green.

---

## One-paragraph recommendation

Go with option (i): a ~915 MB one-time asset dump of the jit module's
state into a sidecar safetensors and a single-header `common/sam3d_body_mhr.h`
that mirrors the 12-stage call graph above. Every op the jit module uses
is plain f32 linear algebra, scalar trig, COO sparse-matvec, or
scatter-add LBS — there is **no** opaque C++ pymomentum kernel to
emulate, just visible Python source. ~14 person-hours, no new
runtime deps, and the structure transfers directly to the future NVRTC
CUDA port. Bring it up stage-by-stage starting with rot6d→rotmat and
ending with the 308-keypoint regression; the local-skel-state stage (7)
is the cheapest non-trivial canary because it's where any quaternion
convention bug will show up first.
