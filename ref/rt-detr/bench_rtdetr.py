#!/usr/bin/env python3
from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor


CKPT = "/mnt/disk01/models/rt_detr_s"
DEFAULT_IMG = "/home/syoyo/work/gemm/main/cpu/sam3d_body/samples/dancing.jpg"
PERSON_LABEL = 0


def ms() -> float:
    return time.perf_counter() * 1000.0


def summarize(name: str, xs: list[float]) -> str:
    if not xs:
        return f"{name}: n/a"
    return (
        f"{name}: mean={statistics.mean(xs):.3f} ms "
        f"min={min(xs):.3f} max={max(xs):.3f}"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=DEFAULT_IMG)
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()

    if args.threads > 0:
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(args.threads)

    t0 = ms()
    proc = RTDetrImageProcessor.from_pretrained(args.ckpt)
    model = RTDetrForObjectDetection.from_pretrained(args.ckpt).eval()
    t_load = ms() - t0

    img = Image.open(args.image).convert("RGB")
    print(f"[bench] image={args.image} size={img.size[0]}x{img.size[1]}")
    print(f"[bench] ckpt={args.ckpt}")
    print(f"[bench] torch={torch.__version__} cuda_available={torch.cuda.is_available()}")
    print(f"[bench] threads={torch.get_num_threads()} interop={torch.get_num_interop_threads()}")
    print(f"[bench] load={t_load:.3f} ms")

    stage_times: dict[str, list[float]] = {
        "backbone": [],
        "encoder": [],
        "decoder": [],
    }
    active: dict[str, float] = {}

    def pre(name: str):
        def hook(_m, _i):
            active[name] = ms()

        return hook

    def post(name: str):
        def hook(_m, _i, _o):
            stage_times[name].append(ms() - active.pop(name, ms()))

        return hook

    handles = [
        model.model.backbone.register_forward_pre_hook(pre("backbone")),
        model.model.backbone.register_forward_hook(post("backbone")),
        model.model.encoder.register_forward_pre_hook(pre("encoder")),
        model.model.encoder.register_forward_hook(post("encoder")),
        model.model.decoder.register_forward_pre_hook(pre("decoder")),
        model.model.decoder.register_forward_hook(post("decoder")),
    ]

    preprocess_times: list[float] = []
    forward_times: list[float] = []
    post_times: list[float] = []
    total_times: list[float] = []
    best = None

    n_total = args.warmup + args.runs
    for run in range(n_total):
        keep = run >= args.warmup
        before_counts = {k: len(v) for k, v in stage_times.items()}
        tt = ms()

        t = ms()
        enc = proc(images=img, return_tensors="pt")
        t_pre = ms() - t

        t = ms()
        with torch.inference_mode():
            out = model(**enc)
        t_forward = ms() - t

        t = ms()
        target_sizes = torch.tensor([img.size[::-1]])
        results = proc.post_process_object_detection(
            out, target_sizes=target_sizes, threshold=args.threshold
        )[0]
        persons = []
        for score, label, box in zip(
            results["scores"], results["labels"], results["boxes"]
        ):
            if int(label) != PERSON_LABEL:
                continue
            x0, y0, x1, y1 = [float(v) for v in box.tolist()]
            persons.append((float(score), x0, y0, x1, y1))
        if persons:
            best = max(persons, key=lambda p: (p[3] - p[1]) * (p[4] - p[2]))
        t_post = ms() - t
        t_total = ms() - tt

        if keep:
            preprocess_times.append(t_pre)
            forward_times.append(t_forward)
            post_times.append(t_post)
            total_times.append(t_total)
        else:
            for k, n in before_counts.items():
                del stage_times[k][n:]

        tag = "warmup" if not keep else f"run{run - args.warmup + 1}"
        print(
            f"[bench] {tag}: total={t_total:.3f} preprocess={t_pre:.3f} "
            f"forward={t_forward:.3f} post={t_post:.3f}"
        )

    for h in handles:
        h.remove()

    print("[bench] summary")
    print("  " + summarize("total", total_times))
    print("  " + summarize("preprocess", preprocess_times))
    print("  " + summarize("forward", forward_times))
    for k in ("backbone", "encoder", "decoder"):
        print("  " + summarize(k, stage_times[k]))
    print("  " + summarize("postprocess", post_times))
    if best:
        print(
            "[bench] largest_person "
            f"score={best[0]:.4f} bbox=({best[1]:.1f},{best[2]:.1f},"
            f"{best[3]:.1f},{best[4]:.1f})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
