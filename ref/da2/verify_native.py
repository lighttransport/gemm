#!/usr/bin/env python3
"""Compare native DA2 stages with exported oracle arrays; requires no Torch."""
import argparse
import json
from pathlib import Path
import resource
import subprocess
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from server.vhuman.reconstruction.depth import _preprocess


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fixture",type=Path,required=True)
    ap.add_argument("--runner",type=Path,required=True)
    ap.add_argument("--backend",choices=("cpu","cuda"),default="cpu")
    args=ap.parse_args()
    root=args.fixture; spec=json.loads((root/"fixture.json").read_text())
    chw,_,_=_preprocess(spec["image"],spec["input_size"])
    np.testing.assert_array_equal(chw.ravel(),np.fromfile(root/"input.f32",dtype="<f4"))
    out=root/("native-"+args.backend);out.mkdir(exist_ok=True)
    cmd=[str(args.runner.resolve()),"--backbone",str(root/"dinov2.safetensors"),
         "--head",str(root/"depth_head.safetensors"),"--input",str(root/"input.f32"),
         "--output",str(out/"depth.f32"),"--dump-dir",str(out),"--backend",args.backend,"--threads","4"]
    for name in ("width","height","output_width","output_height"):
        cmd += ["--"+name.replace("_","-"),str(spec[name])]
    run=subprocess.run(cmd,check=True,capture_output=True,text=True)
    checks={}
    for name in ("feature_0","feature_1","feature_2","feature_3","fused","depth_model","depth"):
        got=np.fromfile(out/(name+".f32"),dtype="<f4")
        ref=np.fromfile(root/("reference.f32" if name=="depth" else "ref_"+name+".f32"),dtype="<f4")
        if got.shape!=ref.shape or not np.isfinite(got).all():
            raise AssertionError("invalid native tensor: "+name)
        error=np.abs(got.astype(np.float64)-ref.astype(np.float64))
        max_gate,mean_gate=(2e-4,2e-5) if name.startswith("depth") else (1e-3,1e-4)
        # The internal head output is relative (arbitrary-scale) depth; allow
        # 1e-4 of its signal peak for accumulated FP32 GEMM roundoff. Keep the
        # final image's absolute 2e-4 gate unchanged for the consumer contract.
        if name=="depth_model":
            max_gate=max(max_gate,float(np.max(np.abs(ref)))*1e-4)
        checks[name]={"max_abs":float(error.max()),"mean_abs":float(error.mean()),
                      "max_gate":max_gate,"mean_gate":mean_gate,
                      "pass":bool(error.max()<max_gate and error.mean()<mean_gate)}
    report={"preprocessing":"exact","checks":checks,"pass":all(x["pass"] for x in checks.values()),
            "runner":json.loads(run.stdout),"peak_host_rss_kib":resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss}
    (out/"parity.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)
    return 0 if report["pass"] else 1


if __name__=="__main__":
    raise SystemExit(main())
