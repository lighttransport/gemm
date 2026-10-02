"""Trainable direct-TTS adapter. No untrained checkpoint is accepted for inference."""
import torch
from torch import nn


class CausalMotion(nn.Module):
    def __init__(self, hidden_size, names, ranges):
        super().__init__()
        if hidden_size < 1 or len(names) != len(set(names)) or not names:
            raise ValueError("invalid adapter configuration")
        self.hidden_size, self.names = hidden_size, tuple(names)
        bounds = torch.as_tensor(ranges, dtype=torch.float32)
        if bounds.shape != (len(names), 2) or not torch.isfinite(bounds).all() or (bounds[:, 0] > bounds[:, 1]).any():
            raise ValueError("invalid adapter ranges")
        self.register_buffer("bounds", bounds)
        self.register_buffer("hidden_mean", torch.zeros(hidden_size))
        self.register_buffer("hidden_scale", torch.ones(hidden_size))
        self.hidden_projection = nn.Linear(hidden_size, 128)
        self.code_embedding = nn.Embedding(2048, 16)
        self.recurrent = nn.GRU(128 + 16 * 16, 128, num_layers=2, batch_first=True)
        self.output = nn.Linear(128, 8 * len(names))

    def forward(self, hidden, codes, state=None):
        if hidden.ndim != 3 or hidden.shape[-1] != self.hidden_size or codes.shape != (*hidden.shape[:2], 16):
            raise ValueError("invalid feature batch")
        normalized = (hidden-self.hidden_mean)/self.hidden_scale
        x = torch.cat((self.hidden_projection(normalized), self.code_embedding(codes).flatten(-2)), -1)
        x, state = self.recurrent(x, state)
        x = self.output(x).reshape(*x.shape[:2], 8, len(self.names))
        low, high = self.bounds[:, 0], self.bounds[:, 1]
        return low + torch.sigmoid(x) * (high - low), state


class ReferenceMotionAdapter:
    def __init__(self, checkpoint, revision, device="cpu", allow_diagnostic=False):
        data = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if data.get("format") != "vhuman.tts_motion.v1" or not data.get("trained") or data.get("tts_revision") != revision:
            raise ValueError("checkpoint is untrained, incompatible, or uses different TTS weights")
        if data.get("purpose") != "production" and not (allow_diagnostic and data.get("purpose") == "diagnostic"):
            raise ValueError("diagnostic motion checkpoint requires explicit diagnostic mode")
        self.model = CausalMotion(data["hidden_size"], data["names"], data["ranges"]).to(device)
        self.text_feed = data.get("text_feed", "incremental")
        state_dict = dict(data["state_dict"])
        # Legacy adapters used raw hidden states; preserve their exact behavior.
        state_dict.setdefault("hidden_mean", self.model.hidden_mean.cpu())
        state_dict.setdefault("hidden_scale", self.model.hidden_scale.cpu())
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.device, self.revision = device, revision
        self.reset(0)

    def reset(self, epoch):
        self.epoch, self.state, self.expected = epoch, None, 0

    def push(self, features):
        from ..pipeline.protocol import MotionFrame
        if features.epoch != self.epoch:
            return []
        if features.model_revision != self.revision or features.sample_start != self.expected:
            raise ValueError("noncontiguous or incompatible TTS features")
        with torch.inference_mode():
            hidden = torch.tensor(features.hidden.copy(), device=self.device)[None, None]
            codes = torch.tensor(features.codes.copy(), device=self.device, dtype=torch.long)[None, None]
            values, self.state = self.model(hidden, codes, self.state)
            values = values[0, 0].cpu().numpy()
        self.expected += 1920
        return [MotionFrame(self.epoch, features.sample_start + i * 240, v) for i, v in enumerate(values)]
