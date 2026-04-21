#!/usr/bin/env python
"""
Dump SAM3 per-stage reference tensors for the CUDA/HIP/CPU verifiers.

Loads the monolithic `sam3.model.safetensors` (strips the `detector_model.`
prefix) with the default transformers `Sam3Config` — image=1008, patch=14,
ViT 32x1024, text 24x1024, FPN 4 levels, DETR enc/dec 6 layers.

Runs the full pipeline on one image + one text phrase and saves every
intermediate that the `verify_*.c` binaries expect into <refdir>.

Usage:
    python gen_image_ref.py [--ckpt PATH] [--image PATH] [--phrase "cat"]
                            [--refdir /tmp/sam3_ref_cat] [--device cuda|cpu]

All outputs are float32 / int64 where appropriate. Tensors are collapsed
to the shape each verifier expects (BCHW / flat / whatever).
"""
import argparse, os, sys
import numpy as np
import torch
from safetensors.torch import load_file
from PIL import Image
from transformers import (
    Sam3Config, Sam3Model, Sam3ImageProcessorFast, Sam3Processor, CLIPTokenizer,
)

PFX = "detector_model."

def load_ckpt(model, ckpt_path):
    sd = load_file(ckpt_path)
    stripped = {(k[len(PFX):] if k.startswith(PFX) else k): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(stripped, strict=False)
    if missing:    print(f"[warn] missing    ({len(missing)}): {missing[:3]}...", file=sys.stderr)
    if unexpected: print(f"[warn] unexpected ({len(unexpected)}): {unexpected[:3]}...", file=sys.stderr)

def save(refdir, name, t):
    if isinstance(t, torch.Tensor): t = t.detach().float().cpu().numpy()
    p = os.path.join(refdir, name + ".npy")
    np.save(p, np.ascontiguousarray(t))
    print(f"  {name:<24s} {t.shape} {t.dtype}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",   default="/mnt/disk01/models/sam3/sam3.model.safetensors")
    ap.add_argument("--image",  default="/home/syoyo/work/gemm/vlm-ptx/cuda/da3/indoor.jpg")
    ap.add_argument("--phrase", default="cat")
    ap.add_argument("--refdir", default="/tmp/sam3_ref_cat")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    os.makedirs(args.refdir, exist_ok=True)
    torch.manual_seed(0)

    print(f"[*] device={args.device} ckpt={args.ckpt} image={args.image} phrase={args.phrase!r}")
    config = Sam3Config()
    model = Sam3Model(config).eval()
    load_ckpt(model, args.ckpt)
    model = model.to(args.device)
    # keep in f32 to give the verifiers the tightest reference.
    model = model.to(torch.float32)

    img = Image.open(args.image).convert("RGB")
    ip = Sam3ImageProcessorFast()
    pv = ip(images=img, return_tensors="pt")["pixel_values"].to(args.device).float()
    tok = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    enc = tok(args.phrase, padding="max_length",
              max_length=config.text_config.max_position_embeddings,
              return_tensors="pt", truncation=True)
    input_ids      = enc["input_ids"].to(args.device)
    attention_mask = enc["attention_mask"].to(args.device)
    # verifiers open these via read_npy_* that accept int64; saved as int64 here.
    save_int64 = lambda name, t: np.save(
        os.path.join(args.refdir, name + ".npy"),
        np.ascontiguousarray(t.detach().cpu().numpy().astype(np.int64)))
    save(args.refdir, "input_pixel_values",  pv[0])
    save_int64("input_input_ids",     input_ids[0])
    save_int64("input_attention_mask",attention_mask[0])

    # ---- Intermediate capture via forward hooks ----
    caps = {}
    def cap(name):
        def hook(_m, _i, o):
            # Accept tuple/list/ModelOutput/Tensor.
            if isinstance(o, (tuple, list)): o = o[0]
            if hasattr(o, "last_hidden_state"): o = o.last_hidden_state
            if not isinstance(o, torch.Tensor):
                caps[name] = o  # keep raw object for custom extraction
                return
            caps[name] = o.detach().float().cpu()
        return hook
    ve = model.vision_encoder
    ve.backbone.embeddings.register_forward_hook(cap("vit_embed"))
    ve.backbone.layers[0].register_forward_hook(cap("vit_block00"))
    ve.backbone.layers[31].register_forward_hook(cap("vit_block31"))
    ve.backbone.register_forward_hook(cap("vision_encoder"))
    for li in range(4):
        ve.neck.fpn_layers[li].register_forward_hook(cap(f"vit_fpn{li}"))

    model.text_encoder.register_forward_hook(cap("text_encoder_raw"))
    model.detr_encoder.register_forward_hook(cap("detr_enc_raw"))
    model.detr_decoder.register_forward_hook(cap("detr_dec_raw"))
    model.mask_decoder.register_forward_hook(cap("mask_dec_raw"))

    # Accumulate dot_product_scoring outputs across the 6 decoder layers.
    dot_per_layer = []
    def _dot_hook(_m, _i, o):
        t = o[0] if isinstance(o, (tuple, list)) else o
        if hasattr(t, "logits"): t = t.logits
        dot_per_layer.append(t.detach().float().cpu().reshape(-1, 200))
    model.dot_product_scoring.register_forward_hook(_dot_hook)

    with torch.no_grad():
        out = model(pixel_values=pv,
                    input_ids=input_ids,
                    attention_mask=attention_mask)

    # ---- Unpack hook outputs ----
    save(args.refdir, "vit_embed",       caps["vit_embed"])
    save(args.refdir, "vit_block00",     caps["vit_block00"])
    save(args.refdir, "vit_block31",     caps["vit_block31"])
    # Backbone forward returns (seq, intermediates?); save flat form.
    save(args.refdir, "vision_encoder",  caps["vision_encoder"])
    for li in range(4):
        save(args.refdir, f"vit_fpn{li}", caps[f"vit_fpn{li}"])

    # text_encoder hook returns BaseModelOutput(last_hidden_state,...)
    te = caps["text_encoder_raw"]
    if hasattr(te, "last_hidden_state"): te = te.last_hidden_state
    save(args.refdir, "text_encoder", te)

    de = caps["detr_enc_raw"]
    if isinstance(de, (tuple, list)): de = de[0]
    save(args.refdir, "detr_enc",     de)
    save(args.refdir, "detr_encoder", de)  # alt name the verifier also looks for

    # Final model output: pred_boxes, pred_masks, pred_logits, etc.
    # Sam3ImageSegmentationOutput fields — introspect.
    for field in ("pred_boxes","pred_masks","pred_logits","presence_logits",
                  "semantic_seg","scores","boxes","masks","final_scores",
                  "final_boxes","final_masks","dot_product_scoring"):
        v = getattr(out, field, None)
        if v is not None:
            save(args.refdir, field, v)

    # Per-layer presence_logits (6,) — verifier reads last layer.
    dd = caps["detr_dec_raw"]
    if hasattr(dd, "presence_logits") and dd.presence_logits is not None:
        pl = dd.presence_logits.detach().float().cpu()  # (6, 1, 1)
        save(args.refdir, "presence_logits", pl.reshape(-1))

    # Per-layer dot_product_scoring captured by hook across the 6 decoder layers.
    if dot_per_layer:
        dot_scores = torch.cat(dot_per_layer, dim=0)  # (6, 200)
        save(args.refdir, "dot_product_scoring", dot_scores)

    # final_* via the processor's post_process
    try:
        proc = Sam3Processor(image_processor=ip, tokenizer=tok)
        # size needed: original PIL size (W, H) -> target_sizes=[(H, W)]
        W, H = img.size
        # Use a low threshold so at least the top-scoring queries survive post-process.
        # The sam3 post-process combines per-query logits with presence_logits, which
        # can push combined scores very low; 0.01 keeps enough candidates for verify.
        res = proc.post_process_instance_segmentation(
            out, threshold=0.01, mask_threshold=0.5,
            target_sizes=[(H, W)])[0]
        # res is a dict: scores, boxes, masks (bool).
        scores = res["scores"].detach().float().cpu()
        boxes  = res["boxes"].detach().float().cpu()
        masks  = res["masks"].detach().cpu().to(torch.uint8)
        save(args.refdir, "final_scores", scores)
        save(args.refdir, "final_boxes",  boxes)
        np.save(os.path.join(args.refdir, "final_masks.npy"),
                np.ascontiguousarray(masks.numpy()))
        print(f"  final_masks            {tuple(masks.shape)} uint8")
    except Exception as e:
        print(f"[warn] final_* post-process failed: {e}")

    print("[ok] wrote ref dumps to", args.refdir)

if __name__ == "__main__":
    main()
