from __future__ import annotations

import re
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .checkpoint import group_name
from .common import memory_guard
from .reap import Saliency


def source_name(name):
    name = name.replace("self_attn.forget_gate.", "self_attn.")
    for site, old in (("attn", "attn"), ("ffn", "ffn")):
        for field in ("fn", "base", "scale"):
            name = name.replace(f"{site}_hc.{field}", f"hc_{old}_{field}")
    return name


def stream_tensor(runtime, name, start, stop, choice, device, dtype):
    if hasattr(runtime, "tensor_weight"):
        return runtime.tensor_weight(name, start, stop, choice, device, dtype)
    return torch.as_tensor(runtime.weight(name, start, stop, choice), device=device, dtype=dtype)


class StreamLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, probabilities, owner):
        ctx.owner = owner
        ctx.save_for_backward(x.detach().cpu(), probabilities.detach().cpu())
        ctx.device = x.device
        ctx.dtype = x.dtype
        selected = int(probabilities.argmax()) if probabilities.numel() else None
        outputs = []
        for start in range(0, owner.rows, owner.runtime.tile):
            stop = min(start+owner.runtime.tile, owner.rows)
            w = stream_tensor(owner.runtime, owner.name, start, stop, selected, x.device, x.dtype)
            outputs.append(F.linear(x, w))
        return torch.cat(outputs, -1)

    @staticmethod
    def backward(ctx, grad):
        owner = ctx.owner
        saved_x, probabilities = ctx.saved_tensors
        x = saved_x.to(ctx.device)
        selected = int(probabilities.argmax()) if probabilities.numel() else None
        dx = torch.zeros_like(x)
        dp = torch.zeros_like(probabilities, device=ctx.device)
        flat_x = x.reshape(-1, x.shape[-1]).float()
        for start in range(0, owner.rows, owner.runtime.tile):
            stop = min(start+owner.runtime.tile, owner.rows)
            g = grad[..., start:stop]
            w = stream_tensor(owner.runtime, owner.name, start, stop, selected, ctx.device, ctx.dtype)
            dx.add_(g @ w)
            if probabilities.numel():
                # Complete task-loss derivative, not reconstruction error.
                dw = g.reshape(-1, stop-start).float().T @ flat_x
                for choice in range(len(probabilities)):
                    candidate = torch.as_tensor(owner.runtime.weight(owner.name, start, stop, choice), device=ctx.device)
                    dp[choice] += (dw*candidate).sum()
        return dx, dp, None


class StreamLinear(nn.Module):
    def __init__(self, runtime, name, rows):
        super().__init__()
        self.runtime, self.name, self.rows = runtime, name, rows

    @property
    def weight(self):
        # The HF indexer uses only weight.dtype to cast its input.
        return torch.empty(0, device=self.runtime.device, dtype=self.runtime.dtype)

    def forward(self, x):
        if self.runtime.capture is not None:
            self.runtime.capture(group_name(self.name), x)
        p = self.runtime.probabilities.get(group_name(self.name), torch.empty(0, device=x.device))
        return StreamLinearFunction.apply(x, p, self)


class StreamEmbeddingFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, probabilities, ids, runtime):
        ctx.runtime, ctx.device = runtime, probabilities.device
        ctx.save_for_backward(ids.cpu(), probabilities.detach().cpu())
        choice = int(probabilities.argmax()) if probabilities.numel() else None
        rows = np.concatenate([runtime.weight("model.language_model.embed_tokens.weight", int(i), int(i)+1, choice) for i in ids.flatten().cpu()])
        return torch.tensor(rows, device=runtime.device, dtype=runtime.dtype).reshape(*ids.shape, runtime.text_config.hidden_size)

    @staticmethod
    def backward(ctx, grad):
        ids, probabilities = ctx.saved_tensors
        dp = torch.zeros_like(probabilities, device=ctx.device)
        flat = grad.reshape(-1, grad.shape[-1]).float()
        for choice in range(len(probabilities)):
            rows = np.concatenate([ctx.runtime.weight("model.language_model.embed_tokens.weight", int(i), int(i)+1, choice) for i in ids.flatten()])
            dp[choice] = (flat*torch.as_tensor(rows, device=ctx.device)).sum()
        return dp, None, None


class StreamExperts(nn.Module):
    def __init__(self, runtime, layer, ids):
        super().__init__()
        self.runtime, self.layer, self.ids = runtime, layer, ids

    def forward(self, hidden, indices, gates):
        result = torch.zeros_like(hidden)
        for local_id in torch.unique(indices).tolist():
            slot, token = torch.where(indices.T == local_id)
            original = self.ids[local_id]
            prefix = f"model.language_model.layers.{self.layer}.mlp.experts.{original}."
            x = hidden[token]
            gate = StreamLinear(self.runtime, prefix+"gate_proj.weight", self.runtime.text_config.moe_intermediate_size)(x)
            up = StreamLinear(self.runtime, prefix+"up_proj.weight", self.runtime.text_config.moe_intermediate_size)(x)
            limit = self.runtime.text_config.swiglu_limit
            intermediate = F.silu(gate.clamp(max=limit))*up.clamp(-limit, limit)
            y = StreamLinear(self.runtime, prefix+"down_proj.weight", self.runtime.text_config.hidden_size)(intermediate)
            if self.runtime.saliency is not None:
                self.runtime.saliency[self.layer].update(original, y, gates[token, slot], torch.ones(len(token), device=y.device, dtype=torch.bool))
            result.index_add_(0, token, (y*gates[token, slot, None]).to(result.dtype))
        return result


class Runtime:
    """Frozen HF attention/mHC with bounded-row weights and CPU saved activations."""
    def __init__(self, checkpoint, config, selected=None, bank=None, device="cpu"):
        from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig
        from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextDecoderLayer, Glm5NextTextRMSNorm
        self.checkpoint, self.config, self.bank = checkpoint, config, bank
        if hasattr(checkpoint, "configure"):
            checkpoint.configure(config, bank.bytes if bank else 0)
            checkpoint.excluded = set(bank.tensors) if bank else set()
        self.device = torch.device(device)
        self.tile = max(config.get("runtime_tile_rows", config["tile_rows"]), 128)
        self.probabilities, self.capture, self.saliency = {}, None, None
        self.dtype = torch.bfloat16
        self.selected = selected or {}
        self.text_config = Glm5NextTextConfig(**checkpoint.config["text_config"])
        self.text_config._attn_implementation = "eager"
        self.layers = []
        for layer in range(self.text_config.num_hidden_layers):
            with torch.device("meta"):
                module = Glm5NextTextDecoderLayer(self.text_config, layer)
            prefix = f"model.language_model.layers.{layer}."
            if hasattr(module.mlp, "experts"):
                ids = self.selected.get(str(layer), list(range(self.text_config.n_routed_experts)))
                module.mlp.experts = StreamExperts(self, layer, ids)
                module.mlp.gate.num_experts = len(ids)
            self._replace(module, prefix)
            for name, parameter in list(module.named_parameters()):
                if not parameter.is_meta:
                    continue
                source = prefix+source_name(name)
                value = checkpoint.read(source)
                if ".mlp.gate.weight" in source and str(layer) in self.selected:
                    value = value[self.selected[str(layer)]]
                parent, field = self._parent(module, name)
                exact = any(part in source for part in (".hc_", ".mlp.gate.", ".A_log", ".dt_bias"))
                parent._parameters[field] = nn.Parameter(torch.tensor(value, device=self.device, dtype=torch.float32 if exact else self.dtype), requires_grad=False)
            for name, buffer in list(module.named_buffers()):
                if buffer is None or not buffer.is_meta:
                    continue
                source = prefix+source_name(name)
                if name == "self_attn.conv1d.weight":
                    value = np.concatenate([checkpoint.read(prefix+f"self_attn.{p}_conv1d.weight") for p in ("q", "k", "v")])
                elif source in checkpoint.tensors:
                    value = checkpoint.read(source)
                    if ".mlp.gate." in source and str(layer) in self.selected:
                        value = value[self.selected[str(layer)]]
                else:
                    raise ValueError(f"Unknown meta buffer {name}; refusing an uninitialized runtime")
                parent, field = self._parent(module, name)
                exact = any(part in source for part in (".mlp.gate.", ".A_log", ".dt_bias"))
                parent._buffers[field] = torch.tensor(value, device=self.device, dtype=torch.float32 if exact else self.dtype)
            module.eval()
            self.layers.append(module)
        self.norm = Glm5NextTextRMSNorm(self.text_config.hidden_size, self.text_config.rms_norm_eps).to(self.device, self.dtype)
        self.norm.weight = nn.Parameter(torch.tensor(checkpoint.read("model.language_model.norm.weight"), device=self.device, dtype=self.dtype), requires_grad=False)
        self.visual = None

    @staticmethod
    def _parent(module, name):
        path, _, field = name.rpartition(".")
        return module.get_submodule(path) if path else module, field

    def _replace(self, module, prefix):
        for name, child in list(module.named_children()):
            full = prefix+name
            if isinstance(child, nn.Linear):
                if child.bias is not None:
                    raise ValueError(f"Unsupported biased lazy linear: {full}")
                setattr(module, name, StreamLinear(self, source_name(full)+".weight", child.out_features))
            elif isinstance(child, nn.Conv1d):
                # Tiny depthwise KDA convolutions are kept resident.
                values = np.concatenate([self.checkpoint.read(prefix+f"{p}_conv1d.weight") for p in ("q", "k", "v")])
                child.weight = nn.Parameter(torch.tensor(values, device=self.device, dtype=self.dtype), requires_grad=False)
            else:
                self._replace(child, full+".")

    def weight(self, name, start, stop, choice=None):
        if self.bank is not None and name in self.bank.tensors:
            return self.bank.read(name, start, stop, choice)
        return self.checkpoint.read(name, start, stop)

    def tensor_weight(self, name, start, stop, choice, device, dtype):
        # Quantized quality readers and RAM-bank candidates must keep their
        # chosen bytes; only unmodified source reads use GPU FP8 decoding.
        source_reader = getattr(self.weight, "__func__", None) is Runtime.weight
        if source_reader and hasattr(self.checkpoint, "tensor") and (self.bank is None or name not in self.bank.tensors):
            return self.checkpoint.tensor(name, start, stop, device, dtype)
        return torch.as_tensor(self.weight(name, start, stop, choice), device=device, dtype=dtype)

    def embedding(self, ids):
        if "model.language_model.embed_tokens.weight" in self.probabilities:
            return StreamEmbeddingFunction.apply(self.probabilities["model.language_model.embed_tokens.weight"], ids, self)
        unique, inverse = torch.unique(ids.cpu(), return_inverse=True)
        values = np.concatenate([self.weight("model.language_model.embed_tokens.weight", int(i), int(i)+1) for i in unique])
        table = torch.tensor(values, device=self.device, dtype=self.dtype)
        return table[inverse.to(self.device)].reshape(*ids.shape, self.text_config.hidden_size)

    def hidden(self, ids, checkpoint=False, vision=None):
        from torch.utils.checkpoint import checkpoint as recompute
        ids = torch.as_tensor(ids, device=self.device, dtype=torch.long).reshape(1, -1)
        hidden = self.embedding(ids)
        if vision is not None:
            hidden = self._vision(hidden, ids, vision)
        hidden = hidden.unsqueeze(2).expand(-1, -1, self.text_config.hc_mult, -1).contiguous()
        mask = torch.ones(ids.shape, device=self.device, dtype=torch.bool)
        positions = torch.arange(ids.shape[1], device=self.device)[None]
        indices = None
        for layer_index, layer in enumerate(self.layers):
            if hasattr(self.checkpoint, "prefetch_layer"):
                self.checkpoint.prefetch_layer(layer_index, getattr(self, "selected", None))
            def block(h, previous, layer=layer):
                return layer(h, attention_mask=mask, position_ids=positions, prev_topk_indices=previous, use_cache=False, chunk_size=self.config.get("kda_chunk_size", 16))
            hidden, indices = recompute(block, hidden, indices, use_reentrant=False) if checkpoint else block(hidden, indices)
            memory_guard(self.config)
            callback = getattr(self, "layer_progress", None)
            if callback is not None:
                callback(layer_index)
        return self.norm(hidden.mean(2))

    def hidden_batch(self, records):
        """Layer-major calibration: share cached weights across CPU-held windows."""
        states = []
        for record in records:
            ids = torch.as_tensor(record[0], device=self.device, dtype=torch.long).reshape(1, -1)
            hidden = self.embedding(ids)
            if len(record) > 2 and record[2] is not None:
                hidden = self._vision(hidden, ids, record[2])
            hidden = hidden.unsqueeze(2).expand(-1, -1, self.text_config.hc_mult, -1).contiguous()
            states.append((hidden.cpu(), None))
        for layer_index, layer in enumerate(self.layers):
            if hasattr(self.checkpoint, "prefetch_layer"):
                self.checkpoint.prefetch_layer(layer_index, getattr(self, "selected", None))
            for index, (hidden, previous) in enumerate(states):
                hidden = hidden.to(self.device)
                previous = previous.to(self.device) if previous is not None else None
                length = hidden.shape[1]
                mask = torch.ones((1, length), device=self.device, dtype=torch.bool)
                positions = torch.arange(length, device=self.device)[None]
                hidden, previous = layer(hidden, attention_mask=mask, position_ids=positions, prev_topk_indices=previous, use_cache=False, chunk_size=self.config.get("kda_chunk_size", 16))
                states[index] = (hidden.cpu(), previous.cpu() if previous is not None else None)
                del hidden, previous
                memory_guard(self.config)
            callback = getattr(self, "layer_progress", None)
            if callback is not None:
                callback(layer_index)
        return [self.norm(hidden.to(self.device).mean(2)).cpu() for hidden, _ in states]

    def _vision(self, hidden, ids, vision):
        from transformers.models.glm5_next.configuration_glm5_next import Glm5NextVisionConfig
        from transformers.models.glm5_next.modeling_glm5_next import Glm5NextVisionModel, Glm5NextVisionRotaryEmbedding
        if self.visual is None:
            with torch.device("meta"):
                model = Glm5NextVisionModel(Glm5NextVisionConfig(**self.checkpoint.config["vision_config"]))
            model.rotary_pos_emb = Glm5NextVisionRotaryEmbedding(model.config.hidden_size // model.config.num_heads // 2).to(self.device)
            values = {name: torch.tensor(self.checkpoint.read("model.visual."+name), dtype=self.dtype, device=self.device) for name in model.state_dict()}
            model.load_state_dict(values, assign=True)
            model.eval().requires_grad_(False)
            self.visual = model
        with torch.no_grad():
            features = self.visual(vision["pixel_values"].to(self.device, self.dtype), grid_thw=vision["image_grid_thw"].to(self.device)).pooler_output
        mask = ids == self.checkpoint.config["image_token_id"]
        if int(mask.sum()) != len(features):
            raise ValueError("Image placeholder count does not match vision features")
        return hidden.masked_scatter(mask[..., None], features)

    def loss(self, ids, answer_mask, checkpoint=False, vision=None):
        hidden = self.hidden(ids, checkpoint, vision)
        labels = torch.as_tensor(ids[1:], device=self.device)
        valid = torch.as_tensor(answer_mask[1:], device=self.device, dtype=torch.bool)
        if not bool(valid.any()):
            raise ValueError("No answer tokens in task window")
        x = hidden[0, :-1][valid]
        labels = labels[valid]
        logits = StreamLinear(self, "lm_head.weight", self.text_config.vocab_size)(x).float()
        return F.cross_entropy(logits, labels)

    def begin_saliency(self):
        self.saliency = {i: Saliency(self.text_config.n_routed_experts) for i in range(self.text_config.num_hidden_layers) if hasattr(self.layers[i].mlp, "experts")}
