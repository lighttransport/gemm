# ref/praxis — PyTorch references for praxis-1-droid frozen parts
Weights in $PRAXIS_ROOT (qwen3.5-2b, wan2.2-ti2v/Wan2.2_VAE.pth, praxis-1-oss clone).
Run with `~/.venv/bin/python`, from repo root; dumps go under `tmp/praxis_ref/`.
- `gen_text_ref.py`: Qwen3.5-2B, conditioning = hidden_states-3] (output of layer 22/24, before final norm, all tokens, no chat template, causal so right-padding is irrelevant).
- `gen_vae_ref.py`: Wan2.2 VAE via praxis/vae.py (48ch latent, 16x spatial / 4x temporal, T=1+4k, H,W%32==0).
- `compare.py`: rel-L2/cosine.
- `gen_real_ref.py`: REAL praxis-1-droid (54 GB) references on a 16 GB GPU via per-block weight streaming (meta-built policy, blocks read from
  safetensors in forward-pre-hooks, bf16, plain PyTorch paths, praxis Session state machine + encode_eager VAE + Qwen3.5 bf16 text). Dumps to
  `tmp/praxis_ref/real/` (see README there). `~/.venv/bin/python -I ref/praxis/gen_real_ref.py` ~5 min; `--selftest <small ckpt>` checks offload==resident.
