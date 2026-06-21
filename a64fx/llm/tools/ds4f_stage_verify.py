#!/usr/bin/env python3
"""Verify a ds4f_stage blob/manifest is byte-identical to the source shards.

For every tensor line in rank<rr>.manifest, look up its source shard via
model.safetensors.index.json, read the exact bytes from that shard (using the
safetensors offset formula 8 + header_len + data_offsets[0]), read the staged
bytes from rank<rr>.blob at the manifest offset, and compare hashes.

Also re-derives the EP ownership rule independently and checks the manifest's
tensor set matches exactly what *should* have been staged for this rank over
the shard range covered (so we catch both corruption and mis-classification).

Usage:
  python3 ds4f_stage_verify.py <stage_dir> <rank> [--model DIR] [--max-shard N] [--sample K]
"""
import sys, os, json, struct, hashlib, argparse

def load_header(path):
    with open(path, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(hlen))
    return hlen, hdr

def tensor_bytes_from_shard(path, hlen, hdr_entry):
    b0, b1 = hdr_entry["data_offsets"]
    abs_off = 8 + hlen + b0
    n = b1 - b0
    with open(path, "rb") as f:
        f.seek(abs_off)
        data = f.read(n)
    if len(data) != n:
        raise IOError("short read %d != %d" % (len(data), n))
    return data

def classify(name, rank, ep_size):
    if name.startswith("mtp."):
        return "skip"
    p = name.find(".experts.")
    if p >= 0:
        q = p + len(".experts.")
        j = q
        while j < len(name) and name[j].isdigit():
            j += 1
        if j > q:
            e = int(name[q:j])
            return "expert" if (e % ep_size == rank) else "skip"
    return "dense"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage_dir")
    ap.add_argument("rank", type=int)
    ap.add_argument("--model", default=os.path.expanduser("~/models/ds4f"))
    ap.add_argument("--max-shard", type=int, default=0,
                    help="only validate set-completeness over shards 1..N (matches DS4F_SHARD_LIMIT)")
    ap.add_argument("--sample", type=int, default=0,
                    help="byte-check only every K-th tensor (0/1 = all)")
    args = ap.parse_args()

    mani = os.path.join(args.stage_dir, "rank%02d.manifest" % args.rank)
    blob = os.path.join(args.stage_dir, "rank%02d.blob" % args.rank)
    idx  = json.load(open(os.path.join(args.model, "model.safetensors.index.json")))
    wmap = idx["weight_map"]

    # parse manifest
    ep_size = None
    entries = []  # (off, nbytes, dtype, ndims, shape, name)
    with open(mani) as f:
        for line in f:
            if line.startswith("#"):
                for tok in line.split():
                    if tok.startswith("ep_size="):
                        ep_size = int(tok.split("=")[1])
                continue
            parts = line.split()
            off = int(parts[0]); nb = int(parts[1]); dtype = parts[2]; nd = int(parts[3])
            shape = [int(x) for x in parts[4:4+nd]]
            name = parts[4+nd]
            entries.append((off, nb, dtype, nd, shape, name))
    assert ep_size is not None, "no ep_size in manifest header"
    print("manifest: %d tensors, ep_size=%d, rank=%d" % (len(entries), ep_size, args.rank))

    bf = open(blob, "rb")
    step = max(1, args.sample)
    n_checked = 0; n_bad = 0; total_bytes = 0
    manifest_names = set()
    for k, (off, nb, dtype, nd, shape, name) in enumerate(entries):
        manifest_names.add(name)
        total_bytes += nb
        # ownership sanity: this rank must legitimately own this tensor
        cls = classify(name, args.rank, ep_size)
        if cls == "skip":
            print("  !! MISCLASSIFIED (should skip): %s" % name); n_bad += 1
        if k % step != 0:
            continue
        shard = wmap.get(name)
        if shard is None:
            print("  !! NOT IN INDEX: %s" % name); n_bad += 1; continue
        spath = os.path.join(args.model, shard)
        hlen, hdr = load_header(spath)
        ref = tensor_bytes_from_shard(spath, hlen, hdr[name])
        if len(ref) != nb:
            print("  !! NBYTES MISMATCH %s: shard=%d manifest=%d" % (name, len(ref), nb)); n_bad += 1; continue
        bf.seek(off)
        got = bf.read(nb)
        if hashlib.sha256(got).digest() != hashlib.sha256(ref).digest():
            print("  !! BYTE MISMATCH: %s (off=%d nb=%d)" % (name, off, nb)); n_bad += 1
        n_checked += 1
    bf.close()

    # set-completeness: over shards 1..max_shard, every owned tensor must be present
    miss = 0
    if args.max_shard > 0:
        want = set()
        for name, shard in wmap.items():
            sidx = int(shard.split("-")[1])  # model-000NN-of-...
            if sidx > args.max_shard:
                continue
            if classify(name, args.rank, ep_size) != "skip":
                want.add(name)
        missing = want - manifest_names
        extra = manifest_names - want
        miss = len(missing) + len(extra)
        print("set-completeness over shards 1..%d: want=%d have=%d missing=%d extra=%d"
              % (args.max_shard, len(want), len(manifest_names), len(missing), len(extra)))
        for m in list(missing)[:8]: print("    missing:", m)
        for e in list(extra)[:8]:   print("    extra:  ", e)

    print("\nbyte-checked %d/%d tensors, %.2f GB total staged" % (n_checked, len(entries), total_bytes/1e9))
    if n_bad == 0 and miss == 0:
        print("RESULT: PASS (byte-identical + ownership correct)")
        return 0
    print("RESULT: FAIL (%d byte/class errors, %d set errors)" % (n_bad, miss))
    return 1

if __name__ == "__main__":
    sys.exit(main())
