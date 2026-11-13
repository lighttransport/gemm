#!/usr/bin/env python3
"""M1 check: replay the reference observation through PraxisClient + praxis_server and compare every stage with the real reference dumps.

  ~/.venv/bin/python -I cuda/praxis/check_client_replay.py [--ref tmp/praxis_ref/real] [--oracle tmp/praxis_ref/real_fp32] [--expert-fmt mixed|nvfp4|bf16]

Same two calls as ref/praxis/gen_real_ref.py: call 1 = stack of 1 frame (depth 0, K'=1, seed 0, 21 rows), call 2 = stack of 22 frames (depth 1, K'=2,
chunk_index 1 -> seed 1, 24 rows), descriptor "" (FR3 hand frame, like the reference). Compared: tile canvas of the encoded window, anchor/clean latents
(VAE), proprio, sampled action tokens, the un-normalised action rows, the commands (pose/grip). The sampled tensors have a large torch-bf16 noise floor:
the torch bf16 reference differs from the torch fp32 oracle by 0.34 (actions, call 1) / ~0.5 (call 2), so the oracle is quoted too.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import praxis_client as pc  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ref", default="tmp/praxis_ref/real")
ap.add_argument("--oracle", default="tmp/praxis_ref/real_fp32")
ap.add_argument("--expert-fmt", default="mixed")
ap.add_argument("--calls", default="1,2")
ap.add_argument("--vae-mode", default="keep")
ap.add_argument("--ckpt", default="/mnt/disk1/praxis/praxis-1-droid")
a = ap.parse_args()
prompt = "pick up the red cup and place it on the plate"


def rel(x, y):
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    return float(np.linalg.norm(x - y) / (np.linalg.norm(y) + 1e-30))


def L(p):
    return np.load(p).astype(np.float32)


env = {"PRAXIS_PACK_CACHE": "/mnt/disk1/praxis/praxis-1-droid.nvfp4.packcache"}
if a.expert_fmt == "bf16":
    env["PRAXIS_BF16_MMAP"] = "1"
else:
    env["PRAXIS_EXPERT_FMT"] = a.expert_fmt
pipe = pc.PraxisClient(a.ckpt, server=None, env=env, vae_mode=a.vae_mode, log_path="/mnt/disk01/tmp/praxis_sim/server_replay.log")
ext, wr, pose, grip = (np.load(f"{a.ref}/obs_{n}.npy") for n in ("exterior", "wrist", "ee_pose", "grip"))
sess = pipe.session(prompt=prompt, seed=0, max_context_chunks=1, control_mode="droid_impedance", exec_horizon=24)
pipe.warm_text([prompt])


def obs(rows):
    return {"ee_pose": pose[:rows], "grip": grip[:rows], "exterior_image": ext[:rows], "wrist_image": wr[:rows], "task": prompt}


for call, rows in ((1, 1), (2, 22)):
    if str(call) not in a.calls.split(","):
        continue
    root = a.ref if call == 2 else f"{a.ref}/call1"
    oroot = a.oracle if call == 2 else f"{a.oracle}/call1"
    res = sess.step(obs(rows))
    print(f"== call {call}: depth {res.depth} seed {res.seed} rows {len(res.commands)} timing {({k: round(v, 3) for k, v in res.timing.items()})}")
    st = sess.state
    # 1. canvas of the encoded window (call 2 only: stream clip) vs the reference tiles
    if call == 2:
        clip = st.stream._clip[0].permute(1, 0, 2, 3).to(np.uint8 if False else __import__("torch").uint8).numpy()
        ref_canvas = np.load(f"{a.ref}/canvas_call2.npy")
        print(f"  canvas window: shape {clip.shape} vs {ref_canvas.shape}, identical pixels: {np.array_equal(clip, ref_canvas)}  max|diff| {np.abs(clip.astype(int) - ref_canvas.astype(int)).max()}")
    # 2. VAE latents
    d = f"{a.ref}/inputs_call{call}"
    anc = res.inputs["anchor"].numpy()
    print(f"  anchor latent rel-L2 vs ref: {rel(anc, L(d + '/anchor.npy')):.3e}")
    if res.inputs["cleans"]:
        print(f"  clean0 latent rel-L2 vs ref: {rel(res.inputs['cleans'][0].numpy(), L(d + '/clean0.npy')):.3e}")
    print(f"  proprio_all rel-L2 vs ref: {rel(res.inputs['proprio_all'].numpy(), L(d + '/proprio_all.npy')):.3e}   boundary poses equal: "
          f"{np.allclose(res.inputs['boundary_poses'], np.load(d + '/boundary_poses.npy'), atol=1e-6)}")
    # 3. sampled tokens / rows (vs torch-bf16 ref and torch-fp32 oracle)
    pa = res.pred_action_norm.numpy()
    for tag, rr in (("torch-bf16 ref", root), ("torch-fp32 oracle", oroot)):
        if os.path.exists(f"{rr}/pred_action.npy"):
            print(f"  pred_action vs {tag}: rel-L2 {rel(pa, L(rr + '/pred_action.npy')):.3e}   action_rel (un-normalised) rel-L2 {rel(res.action_rel, L(rr + '/action_rel.npy')):.3e}")
    if res.pred_video_latent is not None:
        for tag, rr in (("torch-bf16 ref", root), ("torch-fp32 oracle", oroot)):
            if os.path.exists(f"{rr}/pred_video.npy"):
                print(f"  pred_video vs {tag}: rel-L2 {rel(res.pred_video_latent.numpy(), L(rr + '/pred_video.npy')):.3e}")
    # 4. commands: pose12 (xyz + R_flat9) and grip vs ref commands, in the FR3 hand frame; plus the client-frame conversion alone (fed with the ref tokens)
    poses, grips = __import__("praxis.single_arm", fromlist=["x"]).wire_to_arrays(res.commands)
    for tag, rr in (("torch-bf16 ref", root), ("torch-fp32 oracle", oroot)):
        if os.path.exists(f"{rr}/commands_pose.npy"):
            cp, cg = L(rr + "/commands_pose.npy"), L(rr + "/commands_grip.npy")
            print(f"  commands vs {tag}: rows {poses.shape[0]}/{cp.shape[0]}, xyz max|diff| {1e3 * np.abs(poses[:, :3] - cp[:, :3]).max():.2f} mm, "
                  f"R max|diff| {np.abs(poses[:, 3:] - cp[:, 3:]).max():.4f}, grip max|diff| {np.abs(grips - cg).max():.3f}")
    # 5. pure conversion check: reference tokens -> commands through the session's own state (bit-level vs the reference commands)
    import torch
    inp = type("I", (), dict(depth=res.depth, pose8_now=res.inputs["pose_now"]))
    ref_tok = torch.from_numpy(L(root + "/pred_action.npy"))
    wire = st.rel_to_wire(st.decode_actions(inp, ref_tok)[1], res.inputs["pose_now"])
    p2, g2 = __import__("praxis.single_arm", fromlist=["x"]).wire_to_arrays(wire)
    cp, cg = L(root + "/commands_pose.npy"), L(root + "/commands_grip.npy")
    print(f"  CONVERSION ONLY (ref tokens -> commands, session code): rows {p2.shape[0]}, pose max|diff| {np.abs(p2 - cp).max():.3e}, grip max|diff| {np.abs(g2 - cg).max():.3e}, timestamps "
          f"{[round(w['timestamp'], 4) for w in wire[:3]]}...")
print("server info:", pipe.server.info())
pipe.close()
