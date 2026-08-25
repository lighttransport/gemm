#!/usr/bin/env python3
"""Stage the bounded DeepSeek-V4-Flash layer-0 FP4 GEMM weight subset."""
import argparse
import json
import os
import struct
import sys
from pathlib import Path

ALIGN = 256
CHUNK = 8 << 20
MAX_BYTES = 16 << 30

def available_bytes():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) << 10
    raise RuntimeError("MemAvailable is unavailable")

def read_header(path):
    with open(path, "rb", buffering=0) as f:
        size = struct.unpack("<Q", f.read(8))[0]
        return 8 + size, json.loads(f.read(size))

def selected(name):
    if name.startswith("layers.0.attn."):
        return name.endswith(".weight") or name.endswith(".scale")
    if name.startswith("layers.0.ffn.experts."):
        return name.endswith(".weight") or name.endswith(".scale")
    return False

def write_all(fd, data):
    view = memoryview(data)
    while view:
        done = os.write(fd, view)
        if done <= 0:
            raise OSError("short write")
        view = view[done:]

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", type=Path, default=Path.home()/"models/ds4f")
    ap.add_argument("--output-dir", type=Path, default=Path("/local/u14346/ds4f-fp4-gemm"))
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    config = json.loads((args.model_dir/"config.json").read_text())
    if config.get("model_type") != "deepseek_v4" or config.get("expert_dtype") != "fp4":
        raise ValueError("expected DeepSeek-V4 FP4-expert checkpoint")
    index = json.loads((args.model_dir/"model.safetensors.index.json").read_text())["weight_map"]
    names = sorted(n for n in index if selected(n))
    shards = sorted(set(index[n] for n in names))
    if shards != ["model-00002-of-00046.safetensors"]:
        raise ValueError("layer-0 selection unexpectedly spans %r" % shards)
    source = args.model_dir/shards[0]
    data_start, header = read_header(source)
    records=[]
    for name in names:
        rec=header[name]; begin,end=rec["data_offsets"]
        records.append((name,rec["dtype"],rec["shape"],data_start+begin,end-begin))
    total=sum(r[4] for r in records)
    print("DS4F FP4 stage plan: tensors=%d bytes=%d (%.3f GiB) source=%s" %
          (len(records),total,total/(1<<30),source))
    if total > MAX_BYTES: raise MemoryError("selection exceeds 16 GiB")
    if args.plan_only: return
    if available_bytes() < 12 << 30: raise MemoryError("MemAvailable is below 12 GiB guard")
    args.output_dir.mkdir(parents=True,exist_ok=True)
    blob = args.output_dir/"layer0.raw"
    manifest = args.output_dir/"layer0.manifest"
    if blob.exists() and manifest.exists() and not args.force:
        print("stage already exists; use --force to replace"); return
    bt = Path(str(blob)+".tmp")
    mt = Path(str(manifest)+".tmp")
    entries=[]; out=None; src=None
    try:
        src=os.open(source,os.O_RDONLY);out=os.open(bt,os.O_CREAT|os.O_TRUNC|os.O_WRONLY,0o644)
        pos=0
        for name,dtype,shape,source_off,nbytes in records:
            aligned=(pos+ALIGN-1)//ALIGN*ALIGN
            if aligned>pos: write_all(out,b"\0"*(aligned-pos))
            pos=aligned; done=0
            while done<nbytes:
                want=min(CHUNK,nbytes-done); data=os.pread(src,want,source_off+done)
                if len(data)!=want: raise IOError("short read for "+name)
                write_all(out,data);done+=want
                if hasattr(os,"posix_fadvise"):
                    os.posix_fadvise(src,source_off+done-want,want,os.POSIX_FADV_DONTNEED)
            entries.append((pos,nbytes,dtype,shape,name));pos+=nbytes
        os.fsync(out);os.close(out);out=None;os.close(src);src=None
        with open(mt,"w") as f:
            f.write("# offset nbytes dtype ndim dims... name\n")
            for off,nbytes,dtype,shape,name in entries:
                f.write("%d %d %s %d %s %s\n"%(off,nbytes,dtype,len(shape),
                        " ".join(str(x) for x in shape),name))
            f.flush();os.fsync(f.fileno())
        os.replace(bt,blob);os.replace(mt,manifest)
        print("staged blob=%s manifest=%s bytes=%d"%(blob,manifest,pos))
    except Exception:
        if out is not None: os.close(out)
        if src is not None: os.close(src)
        for p in (bt,mt):
            try:p.unlink()
            except FileNotFoundError:pass
        raise

if __name__=="__main__":
    try: main()
    except Exception as exc:
        print("stage_ds4f_fp4: %s"%exc,file=sys.stderr);sys.exit(2)
