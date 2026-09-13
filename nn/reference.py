# SPDX-License-Identifier: MIT
"""MIT-licensed independent PyTorch oracle for the native GN1 network.

Uses only PyTorch mathematical operators and our documented GN1 contract.
No DLshogi/FukauraOu implementation or model weights are imported.
"""
import ctypes as ct
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F


class Config(ct.Structure):
    _fields_ = [(s, ct.c_uint32) for s in (
        "version", "side", "inputs", "actions", "channels", "blocks",
        "attention_every", "head_dim", "value_channels", "value_hidden"
    )] + [("seed", ct.c_uint64), ("memory_limit", ct.c_size_t)]


class Metrics(ct.Structure):
    _fields_ = [("policy", ct.c_float), ("value", ct.c_float),
                ("grad_norm", ct.c_float), ("step", ct.c_uint64)]


class Native:
    def __init__(self, path):
        self.lib = ct.CDLL(str(path))
        L = self.lib
        L.gn_default_config.restype = Config
        L.gn_create.argtypes = [ct.POINTER(Config), ct.c_char_p, ct.c_int]
        L.gn_create.restype = ct.c_void_p
        L.gn_destroy.argtypes = [ct.c_void_p]
        L.gn_error.restype = ct.c_char_p
        L.gn_tensor_count.argtypes = [ct.c_void_p]
        L.gn_tensor_count.restype = ct.c_size_t
        L.gn_tensor.argtypes = [ct.c_void_p, ct.c_size_t, ct.POINTER(ct.c_size_t),
                               ct.POINTER(ct.c_size_t), ct.POINTER(ct.POINTER(ct.c_float)),
                               ct.POINTER(ct.POINTER(ct.c_float))]
        L.gn_tensor.restype = ct.c_char_p
        L.gn_infer.argtypes = [ct.c_void_p, ct.c_size_t, ct.c_void_p, ct.c_void_p, ct.c_void_p]
        L.gn_backward.argtypes = [ct.c_void_p, ct.c_size_t, ct.c_void_p, ct.c_void_p,
                                 ct.c_void_p, ct.POINTER(Metrics)]
        L.gn_update.argtypes = [ct.c_void_p, ct.c_float, ct.c_float, ct.c_float, ct.POINTER(Metrics)]

    def check(self, status):
        if status:
            raise RuntimeError(self.lib.gn_error().decode())

    def tensors(self, model):
        values, grads = {}, {}
        for i in range(self.lib.gn_tensor_count(model)):
            r, c = ct.c_size_t(), ct.c_size_t()
            x, g = ct.POINTER(ct.c_float)(), ct.POINTER(ct.c_float)()
            name = self.lib.gn_tensor(model, i, ct.byref(r), ct.byref(c), ct.byref(x), ct.byref(g)).decode()
            values[name] = np.ctypeslib.as_array(x, shape=(r.value*c.value,)).reshape(r.value, c.value)
            if g:
                grads[name] = np.ctypeslib.as_array(g, shape=(r.value*c.value,)).reshape(r.value, c.value)
        return values, grads


class Reference:
    def __init__(self, config, values, learned):
        self.c = config
        self.p = {k: torch.tensor(v.copy(), requires_grad=k in learned) for k, v in values.items()}

    def linear(self, x, key):
        return F.linear(x, self.p[key+".weight"], self.p[key+".bias"].flatten())

    def conv(self, x, key, kernel):
        w = self.p[key+".weight"].reshape(-1, kernel, kernel, x.shape[1]).permute(0, 3, 1, 2)
        return F.conv2d(x, w, self.p[key+".bias"].flatten(), padding=kernel//2)

    def bn(self, x, key, training):
        return F.batch_norm(x, self.p[key+".mean"].flatten(), self.p[key+".variance"].flatten(),
                            self.p[key+".weight"].flatten(), self.p[key+".bias"].flatten(),
                            training=training, momentum=.1, eps=1e-5)

    def ln(self, x, key):
        return F.layer_norm(x, (x.shape[-1],), self.p[key+".weight"].flatten(), self.p[key+".bias"].flatten(), 1e-5)

    def forward(self, x, training):
        c = self.c
        x = x.permute(0, 3, 1, 2)
        x = F.relu(self.bn(self.conv(x, "stem", 5), "stem_norm", training))
        for block in range(c.blocks):
            key = f"blocks.{block}"
            if c.attention_every and (block+1) % c.attention_every == 0:
                x = x.permute(0, 2, 3, 1).reshape(-1, c.side*c.side, c.channels)
                qkv = self.linear(self.ln(x, key+".norm1"), key+".qkv")
                q, k, v = [t.reshape(t.shape[0], t.shape[1], -1, c.head_dim).transpose(1, 2) for t in qkv.chunk(3, -1)]
                square = torch.arange(c.side*c.side)
                dy = square[:, None]//c.side - square[None, :]//c.side + c.side-1
                dx = square[:, None] % c.side - square[None, :] % c.side + c.side-1
                bias = self.p[key+".attention.relative_bias"][dy*(2*c.side-1)+dx].permute(2, 0, 1)
                prob = (q @ k.transpose(-1, -2) / c.head_dim**.5 + bias).softmax(-1)
                attn = (prob @ v).transpose(1, 2).reshape_as(x)
                x = x + self.linear(attn, key+".proj")
                z = self.ln(x, key+".norm2")
                x = x + self.linear(F.silu(self.linear(z, key+".gate"))*self.linear(z, key+".up"), key+".down")
                x = x.reshape(-1, c.side, c.side, c.channels).permute(0, 3, 1, 2)
            else:
                z = F.relu(self.bn(self.conv(x, key+".conv1", 3), key+".norm1", training))
                x = F.relu(x+self.bn(self.conv(z, key+".conv2", 3), key+".norm2", training))
        tokens = x.permute(0, 2, 3, 1).reshape(-1, c.side*c.side, c.channels)
        policy = self.linear(tokens, "policy").flatten(1)
        value = F.relu(self.linear(tokens, "value.project")).flatten(1)
        value = self.linear(F.relu(self.linear(value, "value.hidden")), "value.output")
        return policy, value


def test(channels=4):
    torch.set_num_threads(1)
    native = Native(Path(__file__).parent / "build/libgn.so")
    c = native.lib.gn_default_config()
    c.side, c.inputs, c.actions, c.channels, c.blocks = 3, 4, 7, channels, 2
    c.attention_every, c.head_dim, c.value_channels, c.value_hidden = 2, 2, 2, 4
    model = native.lib.gn_create(ct.byref(c), b"cpu", 0)
    assert model, native.lib.gn_error()
    try:
        values, gradients = native.tensors(model)
        ref = Reference(c, values, gradients)
        x = np.sin(np.arange(72, dtype=np.float32)*.31).reshape(2, 3, 3, 4)
        policy, value = np.empty((2, 63), np.float32), np.empty((2, 3), np.float32)
        native.check(native.lib.gn_infer(model, 2, x.ctypes.data, policy.ctypes.data, value.ctypes.data))
        rp, rv = ref.forward(torch.from_numpy(x), False)
        np.testing.assert_allclose(policy, rp.detach().numpy(), atol=2e-5, rtol=2e-4)
        np.testing.assert_allclose(value, rv.softmax(-1).detach().numpy(), atol=2e-5, rtol=2e-4)
        target = np.full((2, 63), -1, dtype=np.float32)
        target[:, [1, 3, 17]] = [.3, 0, .7]
        labels = np.array([0, 2], dtype=np.uint32)
        rp, rv = ref.forward(torch.from_numpy(x), True)
        t = torch.from_numpy(target)
        policy_loss = -(t.clamp(min=0)*rp.masked_fill(t < 0, -1e9).log_softmax(-1)).sum(-1).mean()
        value_loss = F.cross_entropy(rv, torch.from_numpy(labels.astype(np.int64)))
        loss = policy_loss + value_loss
        loss.backward()
        metrics = Metrics()
        native.check(native.lib.gn_backward(model, 2, x.ctypes.data, target.ctypes.data, labels.ctypes.data, ct.byref(metrics)))
        np.testing.assert_allclose([metrics.policy, metrics.value], [policy_loss.item(), value_loss.item()], atol=2e-5)
        for name, gradient in gradients.items():
            np.testing.assert_allclose(gradient/2, ref.p[name].grad.numpy(), atol=2e-4, rtol=2e-3, err_msg=name)
        optimizer = torch.optim.AdamW([ref.p[k] for k in gradients], lr=.001, weight_decay=.0001)
        # Validate the optimizer with identical, already-checked gradients.
        # BN shift/key biases have analytically zero gradients; FP32 reduction
        # noise around zero is amplified by Adam's epsilon on its first step.
        for name, gradient in gradients.items():
            ref.p[name].grad = torch.from_numpy(gradient.copy()/2)
        torch.nn.utils.clip_grad_norm_([ref.p[k] for k in gradients], 1)
        optimizer.step()
        native.check(native.lib.gn_update(model, .001, .0001, 1, ct.byref(metrics)))
        for name in gradients:
            np.testing.assert_allclose(values[name], ref.p[name].detach().numpy(), atol=1e-5, rtol=2e-4, err_msg=name)
        print(f"PASS PyTorch C={channels}: forward, losses, all {len(gradients)} parameter gradients, AdamW update")
    finally:
        native.lib.gn_destroy(model)


if __name__ == "__main__":
    test()
    test(32)
