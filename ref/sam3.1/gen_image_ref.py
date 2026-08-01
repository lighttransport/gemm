#!/usr/bin/env python3
"""Dump SAM 3.1 per-stage reference tensors for the CUDA/CPU verifiers.

Unlike sam3, HF transformers has no `Sam3_1` class yet, so this script uses
Meta's public repo (github.com/facebookresearch/sam3, commit from
2026-04-22) via the installed `sam3` package.

Invocation:
    python gen_image_ref.py \\
        --ckpt /mnt/disk01/models/sam3.1/sam3.1_multiplex.pt \\
        --image /home/syoyo/work/gemm/main/fujisan.jpg \\
        --phrase "mountain" \\
        --refdir /tmp/sam3.1_ref \\
        --device cuda

Outputs all tensors as fp32 .npy under --refdir. The C verifiers read them
back as float32 (and int64 for token tensors). Shapes match the CUDA
runner layouts: ViT tokens as (N, D) with N=5184, conv stack outputs as
NCHW with C=256.
"""
import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image


def save(refdir, name, t):
    if isinstance(t, torch.Tensor):
        t = t.detach().float().cpu().numpy()
    p = os.path.join(refdir, name + ".npy")
    np.save(p, np.ascontiguousarray(t))
    print(f"  {name:<28s} {t.shape} {t.dtype}")


def save_int64(refdir, name, t):
    arr = t.detach().cpu().numpy().astype(np.int64)
    np.save(os.path.join(refdir, name + ".npy"), np.ascontiguousarray(arr))
    print(f"  {name:<28s} {arr.shape} {arr.dtype}")


def save_dict_tensors(refdir, name, d):
    for k, v in sorted(d.items()):
        if not isinstance(v, torch.Tensor):
            print(f"  [skip non-tensor] {name}.{k}: {type(v).__name__}")
            continue
        # Keep segmentation-head raw names stable for C verifiers.
        out_name = k if name == "segmentation_head_out" else f"{name}_{k}"
        if v.dtype in (torch.int32, torch.int64):
            save_int64(refdir, out_name, v)
        else:
            save(refdir, out_name, v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",
                    default="/mnt/disk01/models/sam3.1/sam3.1_multiplex.pt")
    ap.add_argument("--image",
                    default="/home/syoyo/work/gemm/main/fujisan.jpg")
    ap.add_argument("--phrase", default="mountain")
    ap.add_argument("--refdir", default="/tmp/sam3.1_ref")
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    os.makedirs(args.refdir, exist_ok=True)
    torch.manual_seed(0)

    print(f"[*] device={args.device}  ckpt={args.ckpt}")
    print(f"    image={args.image}  phrase={args.phrase!r}")

    # Meta's fused MLP kernel hardcodes bf16 internally; override it for
    # reproducible fp32 reference dumps.
    import sam3.perflib.fused as _perflib
    import sam3.model.vitdet as _vitdet
    def _addmm_act_fp32(activation, linear, mat1):
        y = torch.nn.functional.linear(mat1, linear.weight, linear.bias)
        if activation in (torch.nn.functional.gelu, torch.nn.GELU):
            return torch.nn.functional.gelu(y)
        if activation in (torch.nn.functional.relu, torch.nn.ReLU):
            return torch.nn.functional.relu(y)
        raise ValueError(f"Unexpected activation {activation}")
    _perflib.addmm_act = _addmm_act_fp32
    _vitdet.addmm_act = _addmm_act_fp32

    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    model = build_sam3_image_model(
        device=args.device,
        checkpoint_path=args.ckpt,
        load_from_HF=False,
        eval_mode=True,
        enable_segmentation=True,
        enable_inst_interactivity=False,
        compile=False,
    )
    model = model.float()

    processor = Sam3Processor(model, resolution=1008, device=args.device)

    image = Image.open(args.image).convert("RGB")

    # ---- Activation capture via forward hooks --------------------------------
    caps = {}

    def cap(name):
        def hook(_m, _i, o):
            if isinstance(o, (tuple, list)):
                o = o[0]
            if hasattr(o, "tensors"):  # NestedTensor
                o = o.tensors
            if hasattr(o, "last_hidden_state"):
                o = o.last_hidden_state
            if not isinstance(o, torch.Tensor):
                caps[name] = o
                return
            caps[name] = o.detach().float().cpu()
        return hook

    vb = model.backbone.vision_backbone
    trunk = vb.trunk
    trunk.patch_embed.register_forward_hook(cap("patch_embed"))
    def _cap_patch_pos(_m, inputs):
        t = inputs[0] if isinstance(inputs, tuple) else inputs
        if isinstance(t, torch.Tensor):
            caps["patch_pos"] = t.detach().float().cpu()
    trunk.ln_pre.register_forward_pre_hook(_cap_patch_pos)
    trunk.ln_pre.register_forward_hook(cap("ln_pre"))
    n_blocks = len(trunk.blocks)
    for bi in (0, min(15, n_blocks - 1), n_blocks - 1):
        trunk.blocks[bi].register_forward_hook(cap(f"block{bi:02d}"))
    # Final trunk output (post last block, before `convs`).
    trunk.register_forward_hook(cap("trunk_out"))

    for i in range(len(vb.convs)):
        vb.convs[i].register_forward_hook(cap(f"convs{i}"))
    if hasattr(vb, "interactive_convs"):
        for i in range(len(vb.interactive_convs)):
            vb.interactive_convs[i].register_forward_hook(cap(f"interactive{i}"))
    if hasattr(vb, "propagation_convs"):
        for i in range(len(vb.propagation_convs)):
            vb.propagation_convs[i].register_forward_hook(cap(f"propagation{i}"))

    lb = model.backbone.language_backbone.encoder
    n_text = len(lb.transformer.resblocks)
    for li in (0, min(11, n_text - 1), n_text - 1):
        lb.transformer.resblocks[li].register_forward_hook(cap(f"text_block{li:02d}"))
    lb.register_forward_hook(cap("text_encoder_out"))
    lb.ln_final.register_forward_hook(cap("text_ln_final"))

    # Capture tokenized input ids (lb is TextTransformer; forward takes [B, T] int64).
    def _cap_ids(_m, inputs):
        t = inputs[0] if isinstance(inputs, tuple) else inputs
        caps["input_input_ids"] = t.detach().cpu()
    lb.register_forward_pre_hook(_cap_ids)

    if hasattr(model, "transformer") and model.transformer is not None:
        model.transformer.register_forward_hook(cap("transformer_out"))
        if hasattr(model.transformer, "encoder"):
            def _cap_enc(_m, _i, o):
                if isinstance(o, dict):
                    caps["detr_enc_memory"] = o["memory"].detach().float().cpu()
                    if o.get("pos_embed") is not None:
                        caps["detr_enc_pos_embed"] = o["pos_embed"].detach().float().cpu()
                    if o.get("padding_mask") is not None:
                        caps["detr_enc_padding_mask"] = o["padding_mask"].detach().cpu()
                else:
                    caps["detr_enc_memory"] = o[0].detach().float().cpu()
            model.transformer.encoder.register_forward_hook(_cap_enc)
            # Per-encoder-layer output (after transpose to (seq, batch, d)).
            for eli, el in enumerate(model.transformer.encoder.layers):
                def _make(idx):
                    def h(_m, _i, o):
                        t = o if isinstance(o, torch.Tensor) else o[0]
                        caps[f"detr_enc_layer{idx:02d}"] = t.detach().float().cpu()
                    return h
                el.register_forward_hook(_make(eli))
            def _cap_enc_in(_m, inputs, kwargs):
                # TransformerEncoderFusion.forward(src, prompt, src_pos=..., ...)
                args = inputs
                if args:
                    src = args[0]
                    if isinstance(src, list) and isinstance(src[0], torch.Tensor):
                        caps["detr_enc_in_src0"] = src[0].detach().float().cpu()
                    if len(args) >= 2 and isinstance(args[1], torch.Tensor):
                        caps["detr_enc_in_prompt"] = args[1].detach().float().cpu()
                src = kwargs.get("src")
                if isinstance(src, list) and isinstance(src[0], torch.Tensor):
                    caps["detr_enc_in_src0"] = src[0].detach().float().cpu()
                prompt = kwargs.get("prompt")
                if isinstance(prompt, torch.Tensor):
                    caps["detr_enc_in_prompt"] = prompt.detach().float().cpu()
                prompt_mask = kwargs.get("prompt_key_padding_mask")
                if isinstance(prompt_mask, torch.Tensor):
                    caps["detr_enc_in_prompt_mask"] = prompt_mask.detach().cpu()
            model.transformer.encoder.register_forward_pre_hook(
                _cap_enc_in, with_kwargs=True
            )
            # Fusion mode indicator
            if hasattr(model.transformer.encoder, "add_pooled_text_to_img_feat"):
                print("[*] encoder.add_pooled_text_to_img_feat =",
                      model.transformer.encoder.add_pooled_text_to_img_feat)
            enc_l0 = model.transformer.encoder.layers[0]
            print(f"[*] enc layer: pre_norm={enc_l0.pre_norm} "
                  f"pos_attn={enc_l0.pos_enc_at_attn} "
                  f"pos_cq={enc_l0.pos_enc_at_cross_attn_queries} "
                  f"pos_ck={enc_l0.pos_enc_at_cross_attn_keys} "
                  f"act={enc_l0.activation_str}")
            print(f"[*] num enc layers={len(model.transformer.encoder.layers)}")
        if hasattr(model.transformer, "decoder"):
            model.transformer.decoder.register_forward_hook(cap("detr_dec_out"))
    if hasattr(model, "segmentation_head") and model.segmentation_head is not None:
        model.segmentation_head.register_forward_hook(cap("segmentation_head_out"))
    if hasattr(model, "dot_prod_scoring") and model.dot_prod_scoring is not None:
        model.dot_prod_scoring.register_forward_hook(cap("dot_prod_scoring_out"))

    # ---- Run pipeline --------------------------------------------------------
    with torch.inference_mode():
        state = processor.set_image(image)
        output = processor.set_text_prompt(args.phrase, state)

    # ---- Dump raw input (post-transform pixel values) so the verifier can
    #      feed the same numbers to the CUDA runner. Sam3Processor.set_image
    #      doesn't stash the pre-transform tensor, so re-run the transform.
    pv = processor.transform(
        torch.as_tensor(np.array(image)).permute(2, 0, 1).to(args.device)
    ).unsqueeze(0)
    save(args.refdir, "input_pixel_values", pv[0])

    print("[*] captured activations:")
    for name in sorted(caps):
        v = caps[name]
        if isinstance(v, torch.Tensor):
            if v.dtype in (torch.int32, torch.int64):
                save_int64(args.refdir, name, v)
            else:
                save(args.refdir, name, v)
        elif isinstance(v, dict):
            save_dict_tensors(args.refdir, name, v)
        else:
            print(f"  [skip non-tensor] {name}: {type(v).__name__}")

    # ---- Final outputs (masks / boxes / scores) ------------------------------
    if output is not None:
        masks = output.get("masks")
        boxes = output.get("boxes")
        scores = output.get("scores")
        if masks is not None:
            save(args.refdir, "final_masks", masks.to(torch.uint8))
        if boxes is not None:
            save(args.refdir, "final_boxes", boxes)
        if scores is not None:
            save(args.refdir, "final_scores", scores)

    print(f"[ok] wrote ref dumps to {args.refdir}")


if __name__ == "__main__":
    main()
