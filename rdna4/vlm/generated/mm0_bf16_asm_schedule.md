# Generated RDNA4 mm0 BF16 schedule

Tile:
- CTA: 128x128x32
- waves: 4
- wave tile: 64x64
- LDS: 36 KiB total, transposed as kslot*stride+row
- LDS A stride: 144
- LDS B stride: 144

Register constraints:
- accumulators: 16 float8 vectors per wave
- A fragments: a0..a3
- B fragments: b0..b3
- next global fragments: na0..na3, nb0..nb3

Current generated backend:
- HIP source with generated WMMA order and explicit split barriers.
- External symbol: gemm_mm0_bf16_asm

Target handwritten backend:
1. Keep the same ABI and symbol.
2. Replace generated HIP mainloop with GCN:
   - global_load_b128 next A/B fragments early.
   - ds_store_b128 into inactive LDS buffer only after enough load latency.
   - ds_load_b128 small fragment groups.
   - v_wmma_f32_16x16x16_bf16 interleaved with ds_load and stores.
   - s_barrier_signal/wait only at buffer handoff.
3. Preserve 238-ish VGPR budget and zero spills.
