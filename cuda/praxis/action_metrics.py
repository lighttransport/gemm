"""Compare a test_cuda_praxis output dir against a reference dump dir (torch fp32 oracle): rel-L2 of dec_vel / ae_out / sampled video / sampled
actions and the un-normalised action-row errors (rms over the 24 rows: xyz in mm, rot6d, grip), config normalisation of the checkpoint.
  ~/.venv/bin/python -I cuda/praxis/action_metrics.py --ours tmp/praxis_out/x --ref tmp/praxis_ref/real_fp32 [--label name] [--json out.json]
"""
import argparse, json
import numpy as np

CFG = "/mnt/disk1/praxis/praxis-1-droid/config.json"


def rel(a, b):
    a = a.astype(np.float64).ravel(); b = b.astype(np.float64).ravel()
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def unnorm(rows, nz):
    q01 = np.array(nz["action_q01"]); q99 = np.array(nz["action_q99"])
    out = rows.copy().astype(np.float64)
    for d in nz["q99_normed_dims"]:
        n = np.clip(rows[:, d], -1, 1)
        out[:, d] = (n + 1) / 2 * (q99[d] - q01[d]) + q01[d]
    out[:, 9] = np.clip(out[:, 9], 0, 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    nz = json.load(open(CFG))["normalization"]
    res = {}
    for k, f in (("dec_vel", "dec_vel"), ("ae_out", "ae_out"), ("video", "smp_out_video"), ("actions", "smp_out_action")):
        res[k] = rel(np.load(f"{a.ours}/{f}.npy"), np.load(f"{a.ref}/{f}.npy"))
    ro = np.load(f"{a.ours}/smp_out_action.npy").reshape(-1, 10)
    rr = np.load(f"{a.ref}/smp_out_action.npy").reshape(-1, 10)
    uo, ur = unnorm(ro, nz), unnorm(rr, nz)
    d = uo - ur
    res["xyz_mm_rms_err"] = float(np.sqrt((d[:, :3] ** 2).mean()) * 1000)
    res["rot6d_rms_err"] = float(np.sqrt((d[:, 3:9] ** 2).mean()))
    res["grip_rms_err"] = float(np.sqrt((d[:, 9] ** 2).mean()))
    res["xyz_mm_rms_signal"] = float(np.sqrt((ur[:, :3] ** 2).mean()) * 1000)
    print(f"{a.label:34s} dec_vel {res['dec_vel']:.3e} ae_out {res['ae_out']:.3e} video {res['video']:.3f} actions {res['actions']:.3f} | "
          f"xyz {res['xyz_mm_rms_err']:.2f} mm rot6d {res['rot6d_rms_err']:.4f} grip {res['grip_rms_err']:.3f}")
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
