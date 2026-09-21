#!/usr/bin/env python3
"""Check native Euler arithmetic against the official BF16 PyTorch scheduler."""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work-dir", type=Path, default=Path("tmp/qimg21-scheduler-regression"))
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cuda",
                    help="CUDA is the acceptance reference; CPU scalar promotion differs")
    args = ap.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    executable = Path(__file__).with_name("test_scheduler").resolve()
    results = []
    for seed, sigma, next_sigma in [(42,1.0,0.02),(17,0.63,0.37),(123,0.02,0.0)]:
        folder=args.work_dir / str(seed)
        folder.mkdir(exist_ok=True)
        rng=np.random.default_rng(seed)
        sample=torch.from_numpy(rng.standard_normal((256,64),dtype=np.float32)).to(args.device).bfloat16()
        pred=torch.from_numpy(rng.standard_normal((256,64),dtype=np.float32)).to(args.device).bfloat16()
        np.save(folder / "sample.npy",sample.float().cpu().numpy())
        np.save(folder / "prediction.npy",pred.float().cpu().numpy())
        scheduler=FlowMatchEulerDiscreteScheduler(shift=1.0,use_dynamic_shifting=False)
        scheduler.set_timesteps(sigmas=[sigma,next_sigma])
        reference=scheduler.step(pred,scheduler.timesteps[0],sample,return_dict=False)[0].float().cpu().numpy()
        np.save(folder / "reference.npy",reference)
        subprocess.run([str(executable),str(folder / "sample.npy"),str(folder / "prediction.npy"),
                        str(sigma),str(next_sigma),str(folder / "native.npy")],check=True)
        actual=np.load(folder / "native.npy")
        a,b=actual.astype(np.float64).ravel(),reference.astype(np.float64).ravel()
        cosine=float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)))
        result=dict(device=args.device,seed=seed,sigma=sigma,next_sigma=next_sigma,cosine=cosine,
                    exact=bool(np.array_equal(actual,reference)),finite=bool(np.isfinite(actual).all()))
        print(json.dumps(result))
        results.append(result)
    (args.work_dir / "results.json").write_text(json.dumps(results,indent=2)+"\n")
    return 0 if all(r["exact"] and r["finite"] and r["cosine"]>=0.99996 for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
