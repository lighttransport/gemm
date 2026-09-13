"""Record source revisions, checkpoint fingerprints and reference build provenance."""
import argparse
import hashlib
import json
import struct
from collections import Counter
from pathlib import Path
import torch

root=Path(__file__).resolve().parent
p=argparse.ArgumentParser()
p.add_argument("--model-dir",type=Path,default=Path("/mnt/disk2/models/Pixal3D"))
p.add_argument("--dinov3",type=Path,default=Path("/mnt/disk2/models/dinov3-vitl16/model.safetensors"))
p.add_argument("--output",type=Path,required=True)
p.add_argument("--full-hash",action="store_true",help="Stream every checkpoint through SHA256, including tensor payloads")
a=p.parse_args()
pipeline=json.loads((a.model_dir/"pipeline.json").read_text())
files=[a.model_dir/(v+".safetensors") for v in pipeline["args"]["models"].values()]
files += [a.dinov3,root/"weights/naf_release.safetensors"]
manifest={"sources":json.loads((root/"sources.json").read_text()),"torch":torch.__version__,
          "cuda":torch.version.cuda,"hip":torch.version.hip,"checkpoints":[],"pipeline":pipeline}
for path in files:
    with path.open("rb") as f:
        length=struct.unpack("<Q",f.read(8))[0]
        if length>64*1024*1024:raise ValueError(f"Invalid safetensors header in {path}")
        raw=f.read(length)
    header=json.loads(raw)
    tensors={k:v for k,v in header.items() if k!="__metadata__"}
    item={"path":str(path.resolve()),"bytes":path.stat().st_size,
        "header_sha256":hashlib.sha256(raw).hexdigest(),"dtypes":dict(Counter(v["dtype"] for v in tensors.values())),
        "tensor_count":len(tensors)}
    if a.full_hash:
        with path.open("rb") as f:item["sha256"]=hashlib.file_digest(f,"sha256").hexdigest()
    manifest["checkpoints"].append(item)
a.output.parent.mkdir(parents=True,exist_ok=True)
a.output.write_text(json.dumps(manifest,indent=2)+"\n")
print(a.output)
