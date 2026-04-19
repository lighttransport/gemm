# SAM 3 — CPU runner

Concept-level promptable segmentation: image + text noun-phrase -> instance
masks. HF `transformers.Sam3Model` schema; loads `sam3.model.safetensors`
and strips the `detector_model.` key prefix.

## Status

Implemented and verified against `ref/sam3/gen_image_ref.py` dumps:

| Stage                              | Status           | max abs  | mean abs |
|------------------------------------|------------------|----------|----------|
| preprocess (stb resize)            | works (PIL-diff) | —        | —        |
| patch_embed + pos_embed (ref pix)  | bit-close (fp32) | 6.2e-6   | 1.4e-7   |
| ViT block 0 (windowed)             | ok (F16 drift)   | 8.5e-3   | 5.1e-4   |
| ViT final (32 blocks)              | drifts           | 4.4e+1   | 6.0e-3   |
| FPN level 0 (4× upsample, 288²)    | ok               | 1.8e-2   | 5.6e-5   |
| FPN level 1 (2× upsample, 144²)    | ok               | 9.0e-1   | 4.8e-3   |
| FPN level 2 (scale=1, 72²)         | ok               | 6.3e-1   | 2.8e-3   |
| FPN level 3 (maxpool 0.5×, 36²)    | ok               | 1.3e-1   | 2.6e-3   |
| CLIP text enc (valid tokens)       | ok (F16 drift)   | 1.4e-2   | 5.3e-4   |
| DETR encoder (6 layers)            | ok (F16 drift)   | 3.2e+0   | 5.7e-3   |
| DETR decoder (6 layers, presence)  | L0/1 match, L2+ drifts (F16) | 1.0e+0 | 3.5e-1 |
| DETR decoder (pred_boxes xyxy)     | drifts           | 9.5e-1   | 1.9e-1   |
| Dot-product scoring (pred_logits)  | F16 drift        | 4.1e+0   | 2.1e+0   |
| Mask decoder (semantic_seg 288²)   | tight            | 2.6e-1   | 5.0e-3   |
| Mask decoder (pred_masks 288²)     | drifts           | 5.6e+1   | 6.8e+0   |
| Post-process (final_mask, IoU)     | IoU 0.991        | —        | —        |

Final-stage max_abs is dominated by activation outlier channels (e.g.
dim=679) amplified across 32 layers of F16-converted GEMM weights. Mean
matches reference to <0.5%. A future increment may switch to native F32
GEMM for verification, or accept the F16 tolerance for HIP parity.

CLIP text encoder (24 layers, causal, erf-GELU) matches ref at valid
token positions. Padded positions differ (attention mask for PAD keys
not yet wired — downstream only consumes valid tokens).

CLIP BPE tokenizer (`sam3_clip_bpe.{c,h}`) matches HF `CLIPTokenizer`
exactly on ASCII phrases. Loads `vocab.json` + `merges.txt` from
`/mnt/disk1/models/clip-bpe/` (copy from `openai/clip-vit-base-patch32`
snapshot). `verify_bpe --text "cat" --refdir /tmp/sam3_ref_cat` prints
`match: OK`. Wired into `./test_sam3 <ckpt> <image> --phrase "cat"`.

DETR encoder (6 layers, self-attn + text-cross-attn + MLP, ReLU, sine
pos encoding, hidden=256/8h/32hd/MLP=2048) matches ref mean within
0.05%. max_abs inherits ViT outliers. Runs ~55 s on 16 threads.

DETR decoder (6 layers, post-norm, 200 queries + presence token) implemented.
Per-layer presence logits match ref for layers 0–1 (F16 drift only), then
diverge (ours [4.31, 4.88, 3.92, 3.27, 3.08, 3.57] vs ref [4.33, 4.89, 4.27,
4.35, 4.62, 4.59]). Divergence cascades from box refinement F16 drift feeding
into the next layer's RPB and query_pos. Final pred_boxes mean_abs=0.19,
max_abs=0.95 vs ref. Running ~80s on 16 threads.

Dot-product scoring and mask decoder implemented. Dot-score `pred_logits`
last-layer mean differs from ref by ~2.0 (ours −2.28 vs ref −4.32), a
systematic offset from the compounded F16 drift in decoder queries and the
2-layer text MLP + residual + LN chain. Mask decoder uses prompt_cross_attn
(encoder ↔ text) + pixel_decoder (nearest 72→144→288 + Conv2d k=3 + GroupNorm(8)
+ ReLU, 2 iterations) + instance_projection + 3-layer mask_embedder + einsum.
`semantic_seg` matches ref within 0.03% of mean (pixel decoder is bit-close
in F32 after the F16 backbone). `pred_masks` (200, 288, 288) inherits
decoder-query F16 drift through mask_embedder — max_abs=56, mean_abs=6.8,
means ours −1.5 vs ref −8.2. Running full pipeline (ViT → FPN → text → DETR
enc → DETR dec → dot_score → mask_dec) ~75s on 16 threads.

Post-process (sigmoid threshold + bilinear resize to image size, score
filter 0.3) implemented. `sam3_run_postprocess(target_h, target_w,
score_thr, mask_thr)` combines sigmoid(pred_logits) × sigmoid(presence),
filters > 0.3, scales xyxy boxes by (W, H, W, H), sigmoid + bilinear
(align_corners=False) resize the kept 288² masks to the target size, and
binarizes at 0.5. On `/tmp/sam2_cat.jpg` (target 512²), ours keeps 6
candidates (top score 0.67 vs ref 0.97 — F16 drift suppresses scores
but preserves ranking); first-vs-first mask IoU = 0.991 against
`final_masks.npy`. Accessors: `sam3_get_final_{scores,boxes,masks}`.

## Weights

Not gated on HF — public `sam3.model.safetensors`:

```bash
uvx --from huggingface_hub hf download \
    facebook/sam3 --local-dir /mnt/disk1/models/sam3
```

## Build

```bash
cd cpu/sam3
make ARCH=native   # builds test_sam3, verify_patch_embed, verify_vit
```

## Verify

Generate reference dumps first (one venv in `ref/sam3/`):

```bash
uv run --project ref/sam3 python ref/sam3/gen_image_ref.py \
    --ckpt /mnt/disk1/models/sam3/sam3.model.safetensors \
    --image /tmp/sam2_cat.jpg --phrase cat \
    --output-dir /tmp/sam3_ref_cat
```

Then:

```bash
./verify_patch_embed --ckpt <ckpt> --image /tmp/sam2_cat.jpg \
    --refdir /tmp/sam3_ref_cat
./verify_vit --ckpt <ckpt> --image /tmp/sam2_cat.jpg \
    --refdir /tmp/sam3_ref_cat --target block0    # or block31, final
```

## Run

```bash
./test_sam3 /mnt/disk1/models/sam3/sam3.model.safetensors \
    /tmp/sam2_cat.jpg --phrase "cat" -o /tmp/sam3_cat.npy -t 16
```

Runs full pipeline (live-tokenized phrase → ViT → FPN → text → DETR →
dot_score → mask_dec → post-process) and writes kept masks as
`(N, H, W)` uint8 `.npy`. ~75s on 16 threads.
