#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Reference for ja_align: HF wav2vec2 dual-CTC posteriors + black-box aligners.

Builds the model exactly like hiragana-asr's load_checkpoint (Apache-2.0):
Wav2Vec2Model(config of reazon-research/japanese-wav2vec2-large) + kana/phoneme
linear heads, FP32 compute. Dumps (in --dump-dir):
  input.npy         f32 [N]    normalized 16 kHz waveform fed to the model
  w2v_feat.npy      f32 [T,512] CNN features (before feature projection)
  w2v_h0.npy        f32 [T,1024] encoder input after positional conv
  w2v_h12.npy       f32 [T,1024] output of layer `inter_ctc_layer`
  w2v_final.npy     f32 [T,1024] final (layer-normed) encoder output
  phoneme_logp.npy  f32 [T,43]  log-softmax phoneme posteriors
  kana_logp.npy     f32 [T,83]  log-softmax kana posteriors
  greedy.json       greedy CTC phoneme / kana strings
With --phonemes / --kana (space separated symbols), additionally runs
torchaudio.functional.forced_align (black-box oracle) and the ctc-segmentation
package if installed, writing fa_*.npy / ctcseg_*.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

KANA = list("あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをん"
            "がぎぐげござじずぜぞだぢづでどばびぶべぼぱぴぷぺぽぁぃぅぇぉっゃゅょゎー")
PHONEMES = ["A", "E", "I", "N", "O", "U", "a", "b", "by", "ch", "cl", "d", "dy", "e", "f", "g", "gy", "h",
            "hy", "i", "j", "k", "ky", "m", "my", "n", "ny", "o", "p", "py", "r", "ry", "s", "sh", "t", "ts",
            "ty", "u", "v", "w", "y", "z"]


def load_audio(path: str) -> np.ndarray:
    import soundfile as sf
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != 16000:
        import torchaudio.functional as AF
        x = AF.resample(torch.from_numpy(x), sr, 16000).numpy()
    return x


def greedy(ids: np.ndarray, vocab: list[str]) -> list[tuple[str, int, int]]:
    out, prev, start = [], 0, 0
    for t, i in enumerate(ids):
        if i != prev:
            if prev != 0:
                out.append((vocab[prev - 1], start, t))
            start = t
        prev = i
    if prev != 0:
        out.append((vocab[prev - 1], start, len(ids)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/mnt/nvme01/models/speech/japanese-wav2vec2-large-hiragana-ctc/best-medium-ep5-inference.pt")
    ap.add_argument("--config", default="/mnt/nvme01/models/speech/japanese-wav2vec2-large")
    ap.add_argument("--wav", required=True)
    ap.add_argument("--dump-dir", required=True)
    ap.add_argument("--phonemes", default="", help="space-separated phoneme targets for forced alignment")
    ap.add_argument("--kana", default="", help="kana string (no spaces needed) for forced alignment")
    args = ap.parse_args()
    from transformers import Wav2Vec2Config, Wav2Vec2Model

    torch.backends.cuda.matmul.allow_tf32 = False
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    cfg = Wav2Vec2Config.from_pretrained(args.config)
    cfg.mask_time_prob = 0.0
    enc = Wav2Vec2Model(cfg)
    sd = {k[len("encoder."):]: v for k, v in ck["model_state_dict"].items() if k.startswith("encoder.") and "masked_spec_embed" not in k}
    missing, unexpected = enc.load_state_dict(sd, strict=False)
    assert not unexpected, unexpected
    enc = enc.float().eval()
    heads = {n: torch.nn.Linear(1024, ck["model_state_dict"][f"{n}.weight"].shape[0]) for n in ("kana_head", "phoneme_head")}
    for n, h in heads.items():
        h.weight.data = ck["model_state_dict"][f"{n}.weight"].float()
        h.bias.data = ck["model_state_dict"][f"{n}.bias"].float()
    inter = int(ck.get("inter_ctc_layer", 12))

    x = load_audio(args.wav)
    xn = (x - x.mean()) / np.sqrt(x.var() + 1e-7)  # Wav2Vec2FeatureExtractor do_normalize
    xn = xn.astype(np.float32)
    out = Path(args.dump_dir)
    out.mkdir(parents=True, exist_ok=True)
    feats = {}
    def feat_hook(_m, _i, o):
        feats.setdefault("feat", o[0].T.numpy())

    enc.feature_extractor.register_forward_hook(feat_hook)
    with torch.no_grad():
        r = enc(torch.from_numpy(xn)[None], output_hidden_states=True)
        hs = r.hidden_states
        final = r.last_hidden_state[0]
        ph = torch.log_softmax(heads["phoneme_head"](hs[inter][0]), -1)
        ka = torch.log_softmax(heads["kana_head"](final), -1)
    np.save(out / "input.npy", xn)
    np.save(out / "w2v_feat.npy", feats["feat"].astype(np.float32))
    np.save(out / "w2v_h0.npy", hs[0][0].numpy())
    np.save(out / f"w2v_h{inter}.npy", hs[inter][0].numpy())
    np.save(out / "w2v_final.npy", final.numpy())
    np.save(out / "phoneme_logp.npy", ph.numpy())
    np.save(out / "kana_logp.npy", ka.numpy())
    gp = greedy(ph.argmax(-1).numpy(), PHONEMES)
    gk = greedy(ka.argmax(-1).numpy(), KANA)
    res = {"frames": int(ph.shape[0]), "frame_sec": 0.02,
           "phonemes": " ".join(p for p, _, _ in gp), "kana": "".join(k for k, _, _ in gk),
           "phoneme_segments": gp, "kana_segments": gk}
    (out / "greedy.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(res["phonemes"])
    print(res["kana"])

    targets = []
    if args.phonemes:
        targets.append(("phoneme", ph, [PHONEMES.index(p) + 1 for p in args.phonemes.split()]))
    if args.kana:
        targets.append(("kana", ka, [KANA.index(c) + 1 for c in args.kana.replace(" ", "")]))
    for name, logp, tgt in targets:
        np.save(out / f"fa_{name}_targets.npy", np.asarray(tgt, dtype=np.int32))
        try:
            import torchaudio.functional as AF
            ali, scores = AF.forced_align(logp[None], torch.tensor([tgt], dtype=torch.int32), blank=0)
            np.save(out / f"fa_{name}_path.npy", ali[0].numpy().astype(np.int32))
            np.save(out / f"fa_{name}_scores.npy", scores[0].numpy().astype(np.float32))
            print(f"torchaudio forced_align({name}): ok, total logp={float(scores[0].sum()):.4f}")
        except Exception as e:  # noqa: BLE001
            print(f"torchaudio forced_align unavailable: {e}")
        try:
            import ctc_segmentation as cs
            vocab = ["<blank>"] + (PHONEMES if name == "phoneme" else KANA)
            cfgs = cs.CtcSegmentationParameters()
            cfgs.char_list = vocab
            cfgs.index_duration = 0.02
            cfgs.blank = 0
            gt = np.asarray([tgt], dtype=np.int64)  # one utterance
            ground_truth_mat, utt_begin_indices = cs.prepare_token_list(cfgs, [np.asarray(tgt)])
            timings, char_probs, state_list = cs.ctc_segmentation(cfgs, logp.numpy(), ground_truth_mat)
            np.save(out / f"ctcseg_{name}_timings.npy", np.asarray(timings, dtype=np.float32))
            np.save(out / f"ctcseg_{name}_charprobs.npy", np.asarray(char_probs, dtype=np.float32))
            segs = cs.determine_utterance_segments(cfgs, utt_begin_indices, char_probs, timings, ["utt"])
            (out / f"ctcseg_{name}.json").write_text(json.dumps({"segments": segs, "gt_shape": list(gt.shape)}))
            print(f"ctc_segmentation({name}): {segs}")
        except Exception as e:  # noqa: BLE001
            print(f"ctc_segmentation unavailable: {e}")


if __name__ == "__main__":
    main()
