"""
Debug: feed our_layer0_pose_raw.bin (519,) through the BODY MHRHead path
(model.head_pose) and compare j3d output (post-/100, post-flip) against
decoder_pose_layer0__pred_keypoints_3d.npy.

If there's a mismatch here, the bug is in MHRHead.forward decoding —
either in our pose decode (decode_pose_raw), or in our mhr_forward,
or in our keypoints_from_mesh.

Usage:
    python debug_layer0_kp3d.py --local-ckpt-dir /mnt/disk01/models/sam3d-body/dinov3 \
        --refdir /tmp/sam3d_body_ref
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch


def _to_np(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().float().numpy()
    return np.asarray(x)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--local-ckpt-dir",
                    default="/mnt/disk01/models/sam3d-body/dinov3")
    ap.add_argument("--refdir", default="/tmp/sam3d_body_ref")
    args = ap.parse_args()

    for cand in ("/tmp/sam-3d-body", os.environ.get("SAM_3D_BODY_DIR", "")):
        if cand and os.path.isdir(cand) and cand not in sys.path:
            sys.path.insert(0, cand)
    try:
        from sam_3d_body import load_sam_3d_body  # type: ignore
    except Exception as e:
        print("cannot import sam_3d_body. Set SAM_3D_BODY_DIR or restore "
              "/tmp/sam-3d-body with: git clone --depth 1 "
              "https://github.com/facebookresearch/sam-3d-body "
              "/tmp/sam-3d-body",
              file=sys.stderr)
        raise e

    ckpt = os.path.join(args.local_ckpt_dir, "model.ckpt")
    mhr_path = os.path.join(args.local_ckpt_dir, "assets", "mhr_model.pt")
    model, _cfg = load_sam_3d_body(checkpoint_path=ckpt, device="cpu",
                                   mhr_path=mhr_path)
    model.eval()
    head_pose = model.head_pose       # body branch (enable_hand_model=False)
    init_pose = model.init_pose.weight  # (1, 519)

    # Load our layer-0 pose_raw (PRE-add init_pose).
    pose_raw = np.fromfile(
        os.path.join(args.refdir, "our_layer0_pose_raw.bin"), dtype=np.float32)
    print(f"pose_raw shape: {pose_raw.shape}")

    # Reference: feed pose_raw + init_pose into head_pose via its public API.
    # But head_pose.forward expects a pose_token (B, 1024) and goes proj→pred.
    # We already have pred_raw = pose_raw, so manually replicate post-proj path.
    pred_raw = torch.from_numpy(pose_raw).unsqueeze(0)  # (1, 519)
    init = init_pose.detach()                          # (1, 519)
    pred = pred_raw + init                             # (1, 519)

    # Replicate MHRHead.forward post-proj (lines 288–367 of mhr_head.py):
    # NOTE: roma is needed for euler conversions. Import after sam_3d_body
    # so its registered ops are available.
    import roma                                       # type: ignore
    from sam_3d_body.models.heads.mhr_head import (    # type: ignore
        rot6d_to_rotmat, compact_cont_to_model_params_body,
        mhr_param_hand_mask)

    count = 6
    global_rot_6d = pred[:, :count]
    global_rot_rotmat = rot6d_to_rotmat(global_rot_6d)
    global_rot_euler = roma.rotmat_to_euler("ZYX", global_rot_rotmat)
    global_trans = torch.zeros_like(global_rot_euler)
    print(f"global_rot_euler: {global_rot_euler.numpy().ravel()}")
    print(f"global_trans:     {global_trans.numpy().ravel()}")

    pred_pose_cont = pred[:, count: count + head_pose.body_cont_dim]
    count += head_pose.body_cont_dim
    pred_pose_euler = compact_cont_to_model_params_body(pred_pose_cont)
    pred_pose_euler[:, mhr_param_hand_mask] = 0
    pred_pose_euler[:, -3:] = 0

    pred_shape = pred[:, count: count + head_pose.num_shape_comps]
    count += head_pose.num_shape_comps
    pred_scale = pred[:, count: count + head_pose.num_scale_comps]
    count += head_pose.num_scale_comps
    pred_hand = pred[:, count: count + head_pose.num_hand_comps * 2]
    count += head_pose.num_hand_comps * 2
    pred_face = pred[:, count: count + head_pose.num_face_comps] * 0
    count += head_pose.num_face_comps
    print(f"count after slicing = {count}, expected = {pred.shape[1]}")
    print(f"pred_shape: norm={pred_shape.norm().item():.4f}")
    print(f"pred_scale: norm={pred_scale.norm().item():.4f}")
    print(f"pred_hand:  norm={pred_hand.norm().item():.4f}")

    # Now mhr_forward.
    output = head_pose.mhr_forward(
        global_trans=global_trans,
        global_rot=global_rot_euler,
        body_pose_params=pred_pose_euler,
        hand_pose_params=pred_hand,
        scale_params=pred_scale,
        shape_params=pred_shape,
        expr_params=pred_face,
        do_pcblend=True,
        return_keypoints=True,
        return_joint_coords=True,
        return_model_params=True,
        return_joint_rotations=True,
    )
    verts, j3d, jcoords, mhr_model_params, joint_global_rots = output
    print(f"verts shape: {verts.shape}, j3d shape: {j3d.shape}, "
          f"jcoords shape: {jcoords.shape}")
    j3d70 = j3d[:, :70]
    # Camera flip
    verts_flipped = verts.clone()
    verts_flipped[..., [1, 2]] *= -1
    j3d70_flipped = j3d70.clone()
    j3d70_flipped[..., [1, 2]] *= -1

    # Compare against the reference layer0 keypoints_3d.
    ref_kp3d = np.load(
        os.path.join(args.refdir, "decoder_pose_layer0__pred_keypoints_3d.npy"))
    ref_kp3d = ref_kp3d.reshape(70, 3)
    ours_recon = _to_np(j3d70_flipped).reshape(70, 3)
    diff = np.abs(ours_recon - ref_kp3d)
    print(f"\n=== ref vs ours-recon (python pipeline) ===")
    print(f"max_abs={diff.max():.4e}  mean_abs={diff.mean():.4e}")
    idx = np.unravel_index(diff.argmax(), diff.shape)
    print(f"argmax at idx={idx}: ours={ours_recon[idx]:.4f}  "
          f"ref={ref_kp3d[idx]:.4f}")

    # Compare against our C output.
    c_kp3d = np.fromfile(
        os.path.join(args.refdir, "our_layer0_kp3d.bin"),
        dtype=np.float32).reshape(70, 3)
    diff2 = np.abs(c_kp3d - ref_kp3d)
    print(f"\n=== ref vs C ===")
    print(f"max_abs={diff2.max():.4e}  mean_abs={diff2.mean():.4e}")
    idx2 = np.unravel_index(diff2.argmax(), diff2.shape)
    print(f"argmax at idx={idx2}: c={c_kp3d[idx2]:.4f}  "
          f"ref={ref_kp3d[idx2]:.4f}")

    diff3 = np.abs(c_kp3d - ours_recon)
    print(f"\n=== C vs ours-recon ===")
    print(f"max_abs={diff3.max():.4e}  mean_abs={diff3.mean():.4e}")

    # Print 5 keypoints side by side.
    print("\nidx |   C[0]   |   C[1]   |   C[2]   |  py[0]   |  py[1]   |  py[2]   |  ref[0]  |  ref[1]  |  ref[2]")
    for k in [0, 10, 20, 30, 50, 54, 60, 69]:
        print(f"{k:3d} | {c_kp3d[k,0]:8.4f} | {c_kp3d[k,1]:8.4f} | "
              f"{c_kp3d[k,2]:8.4f} | {ours_recon[k,0]:8.4f} | "
              f"{ours_recon[k,1]:8.4f} | {ours_recon[k,2]:8.4f} | "
              f"{ref_kp3d[k,0]:8.4f} | {ref_kp3d[k,1]:8.4f} | "
              f"{ref_kp3d[k,2]:8.4f}")

    # Also save mhr_model_params for our C-side debug.
    mp = _to_np(mhr_model_params).ravel()
    print(f"\nmhr_model_params shape={mp.shape}")
    print(f"  global_trans (first 3): {mp[0:3]}")
    print(f"  global_rot   (next 3):  {mp[3:6]}")
    print(f"  full_pose tail (-10:):  {mp[-10:]}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
