#!/usr/bin/env python3
"""Stage one real Kimi K3 KDA head without mapping full tensors."""
import argparse
import json
import os
import sys
from pathlib import Path

import k3_stage

ALIGN = 256
HIDDEN = 7168
HEAD_DIM = 128
HEADS = 96


def specs(layer, head):
    base = "language_model.model.layers.%d.self_attn." % layer
    row0 = head * HEAD_DIM
    return [
        (base+"q_proj.weight", "rows", row0, HEAD_DIM),
        (base+"k_proj.weight", "rows", row0, HEAD_DIM),
        (base+"v_proj.weight", "rows", row0, HEAD_DIM),
        (base+"g_proj.weight", "rows", row0, HEAD_DIM),
        (base+"f_a_proj.weight", "full", 0, 0),
        (base+"f_b_proj.weight", "rows", row0, HEAD_DIM),
        (base+"b_proj.weight", "rows", head, 1),
        (base+"q_conv1d.weight", "rows", row0, HEAD_DIM),
        (base+"k_conv1d.weight", "rows", row0, HEAD_DIM),
        (base+"v_conv1d.weight", "rows", row0, HEAD_DIM),
        (base+"dt_bias", "vector", row0, HEAD_DIM),
        (base+"A_log", "full", 0, 0),
        (base+"o_norm.weight", "full", 0, 0),
    ]


def locate(model_dir, layer, head):
    wanted = {x[0]: x[1:] for x in specs(layer, head)}
    found = {}
    paths = sorted(model_dir.glob("*.safetensors"))
    if len(paths) != 96:
        raise ValueError("expected 96 shards, found %d" % len(paths))
    for path in paths:
        data_start, header = k3_stage.read_header(path)
        for name in set(wanted).intersection(header):
            info = header[name]
            begin, end = info["data_offsets"]
            mode, start, count = wanted[name]
            shape = info["shape"]
            dtype = info["dtype"]
            item = {"BF16": 2, "F32": 4}.get(dtype)
            if item is None:
                raise ValueError("unsupported dtype %s for %s" % (dtype, name))
            if mode == "full":
                byte_start, nbytes, staged_shape = 0, end-begin, shape
            elif mode == "vector":
                if len(shape) != 1 or start+count > shape[0]:
                    raise ValueError("bad vector slice for %s" % name)
                byte_start, nbytes, staged_shape = start*item, count*item, [count]
            else:
                if len(shape) < 2 or start+count > shape[0]:
                    raise ValueError("bad row slice for %s" % name)
                row_elems = 1
                for dim in shape[1:]: row_elems *= dim
                byte_start = start*row_elems*item
                nbytes = count*row_elems*item
                staged_shape = [count] + shape[1:]
            if byte_start+nbytes > end-begin:
                raise ValueError("slice exceeds %s" % name)
            found[name] = {"name": name, "source": str(path),
                           "source_offset": data_start+begin+byte_start,
                           "nbytes": nbytes, "dtype": dtype, "shape": staged_shape}
    missing = sorted(set(wanted)-set(found))
    if missing: raise ValueError("missing tensors: %s" % ", ".join(missing))
    return [found[name] for name,_,_,_ in specs(layer,head)]


def write_all(fd, data):
    view = memoryview(data)
    while view:
        n = os.write(fd, view)
        if n <= 0: raise IOError("short write")
        view = view[n:]


def stage(records, output_dir, layer, head, chunk_bytes, force):
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = "layer%02d_head%02d" % (layer, head)
    blob, manifest = output_dir/(stem+".blob"), output_dir/(stem+".manifest")
    if not force and (blob.exists() or manifest.exists()):
        raise FileExistsError("output exists; pass --force")
    suffix = ".tmp.%d" % os.getpid()
    btmp, mtmp = Path(str(blob)+suffix), Path(str(manifest)+suffix)
    fd = None
    try:
        fd = os.open(str(btmp),os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
        offset=0; entries=[]
        for rec in records:
            aligned=(offset+ALIGN-1)&~(ALIGN-1)
            if aligned>offset: write_all(fd,b"\0"*(aligned-offset))
            src=os.open(rec["source"],os.O_RDONLY)
            try:
                k3_stage.copy_range(src,fd,rec["source_offset"],rec["nbytes"],chunk_bytes)
            finally: os.close(src)
            ent=dict(rec); ent["offset"]=aligned; entries.append(ent)
            offset=aligned+rec["nbytes"]
        os.fsync(fd); os.close(fd); fd=None
        with mtmp.open("w") as f:
            f.write("# K3KDAHEADV1 layer=%d head=%d tensors=%d blob_bytes=%d\n" %
                    (layer,head,len(entries),offset))
            for e in entries:
                f.write("%d %d %s %d %s %s\n" %
                        (e["offset"],e["nbytes"],e["dtype"],len(e["shape"]),
                         " ".join(str(x) for x in e["shape"]),e["name"]))
            f.flush(); os.fsync(f.fileno())
        os.replace(str(btmp),str(blob)); os.replace(str(mtmp),str(manifest))
        return blob,manifest,offset
    except Exception:
        if fd is not None: os.close(fd)
        for p in (btmp,mtmp):
            try: p.unlink()
            except FileNotFoundError: pass
        raise


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir",type=Path,default=Path.home()/"models/kimi-k3")
    ap.add_argument("--output-dir",type=Path,required=True)
    ap.add_argument("--layer",type=int,default=0)
    ap.add_argument("--head",type=int,default=0)
    ap.add_argument("--chunk-mib",type=int,default=8)
    ap.add_argument("--force",action="store_true")
    args=ap.parse_args()
    if args.layer<0 or args.layer>92: ap.error("layer must be in [0,92]")
    if args.head<0 or args.head>=HEADS: ap.error("head must be in [0,95]")
    if k3_stage.mem_available_kb()<k3_stage.MIN_AVAILABLE_KB:
        raise MemoryError("MemAvailable is below 6 GiB")
    records=locate(args.model_dir,args.layer,args.head)
    total=sum(r["nbytes"] for r in records)
    print("K3 KDA head plan: layer=%d head=%d tensors=%d payload=%.3f MiB" %
          (args.layer,args.head,len(records),total/1048576.0))
    blob,manifest,size=stage(records,args.output_dir,args.layer,args.head,
                             args.chunk_mib*1048576,args.force)
    print("staged %.3f MiB: %s" % (size/1048576.0,blob))
    print("manifest: %s" % manifest)


if __name__=="__main__":
    try: main()
    except Exception as exc:
        print("k3_kda_stage: %s" % exc,file=sys.stderr); sys.exit(2)
