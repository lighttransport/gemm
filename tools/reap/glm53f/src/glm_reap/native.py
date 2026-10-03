from __future__ import annotations

import ctypes
import math
import numpy as np

FORMATS = {"F32": (0, 1, 4), "F16": (1, 1, 2), "Q8_0": (8, 32, 34), "Q2_K": (10, 256, 84), "Q3_K": (11, 256, 110), "Q4_K": (12, 256, 144), "Q6_K": (14, 256, 210)}


def nbytes(shape, qtype):
    _, block, size = FORMATS[qtype]
    if shape[-1] % block:
        raise ValueError(f"{qtype} requires rows divisible by {block}: {shape}")
    return math.prod(shape) // block * size


class NativeCodec:
    def __init__(self, library):
        self.lib = ctypes.CDLL(str(library))
        self.lib.ggml_quantize_chunk.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
        self.lib.ggml_quantize_chunk.restype = ctypes.c_size_t

    def encode(self, weight, qtype, importance=None):
        weight = np.ascontiguousarray(weight, dtype=np.float32)
        if not np.isfinite(weight).all():
            raise ValueError("Cannot quantize non-finite weights")
        if qtype in ("F16", "F32"):
            return weight.astype(np.float16 if qtype == "F16" else np.float32).view(np.uint8).reshape(-1).copy()
        result = np.empty(nbytes(weight.shape, qtype), dtype=np.uint8)
        importance = None if importance is None else np.ascontiguousarray(importance, dtype=np.float32)
        if importance is not None and importance.shape != (weight.shape[-1],):
            raise ValueError("Importance must have one entry per input column")
        written = self.lib.ggml_quantize_chunk(FORMATS[qtype][0], weight.ctypes.data, result.ctypes.data, 0, math.prod(weight.shape[:-1]), weight.shape[-1], None if importance is None else importance.ctypes.data)
        if written != result.nbytes:
            raise RuntimeError(f"Native quantizer wrote {written}, expected {result.nbytes}")
        return result


def unpack(raw, qtype):
    """Return native integer codes, scale fields, and superblock scales."""
    _, block, size = FORMATS[qtype]
    b = np.asarray(raw, dtype=np.uint8).reshape(-1, size)
    count = len(b)
    if qtype == "Q8_0":
        return b[:, 2:].view(np.int8).astype(np.int16), np.ones((count, 1), np.int16), b[:, :2].copy().view(np.float16).astype(np.float32), None, None
    if qtype == "Q2_K":
        fields, qs = b[:, :16], b[:, 16:80]
        q = ((qs.reshape(count, 2, 1, 32) >> np.arange(0, 8, 2, dtype=np.uint8)[None, None, :, None]) & 3).reshape(count, block)
        return q.astype(np.int16), (fields & 15).astype(np.int16), b[:, 80:82].copy().view(np.float16).astype(np.float32), (fields >> 4).astype(np.int16), b[:, 82:84].copy().view(np.float16).astype(np.float32)
    if qtype == "Q3_K":
        h, qs, s = b[:, :32], b[:, 32:96], b[:, 96:108]
        lo = ((qs.reshape(count, 2, 1, 32) >> np.arange(0, 8, 2, dtype=np.uint8)[None, None, :, None]) & 3).reshape(count, block)
        hi = ((h[:, None, :] >> np.arange(8, dtype=np.uint8)[None, :, None]) & 1).reshape(count, block)
        sl = ((s[:, :8, None].transpose(0, 2, 1) >> np.array([0, 4], dtype=np.uint8)[None, :, None]) & 15).reshape(count, 16)
        sh = ((s[:, 8:12, None].transpose(0, 2, 1) >> np.arange(0, 8, 2, dtype=np.uint8)[None, :, None]) & 3).reshape(count, 16)
        return lo.astype(np.int16) - (1-hi.astype(np.int16))*4, (sl | (sh << 4)).astype(np.int16)-32, b[:, 108:110].copy().view(np.float16).astype(np.float32), None, None
    if qtype == "Q4_K":
        s = b[:, 4:16]
        scales = np.concatenate((s[:, :4] & 63, (s[:, 8:12] & 15) | ((s[:, :4] >> 2) & 48)), axis=1)
        mins = np.concatenate((s[:, 4:8] & 63, (s[:, 8:12] >> 4) | ((s[:, 4:8] >> 2) & 48)), axis=1)
        q = ((b[:, 16:].reshape(count, 4, 1, 32) >> np.array([0, 4], dtype=np.uint8)[None, None, :, None]) & 15).reshape(count, block)
        return q.astype(np.int16), scales.astype(np.int16), b[:, :2].copy().view(np.float16).astype(np.float32), mins.astype(np.int16), b[:, 2:4].copy().view(np.float16).astype(np.float32)
    if qtype == "Q6_K":
        lo = ((b[:, :128].reshape(count, 2, 1, 64) >> np.array([0, 4], dtype=np.uint8)[None, None, :, None]) & 15).reshape(count, block)
        hi = ((b[:, 128:192].reshape(count, 2, 1, 32) >> np.arange(0, 8, 2, dtype=np.uint8)[None, None, :, None]) & 3).reshape(count, block)
        return (lo | (hi << 4)).astype(np.int16)-32, b[:, 192:208].view(np.int8).astype(np.int16), b[:, 208:210].copy().view(np.float16).astype(np.float32), None, None
    raise ValueError(qtype)


def pack(qtype, codes, scales, d, mins=None, dmin=None):
    q = np.asarray(codes, dtype=np.int16)
    count = len(q)
    _, block, size = FORMATS[qtype]
    out = np.zeros((count, size), np.uint8)
    fp16 = lambda x: np.ascontiguousarray(x, dtype=np.float16).reshape(count, 1).view(np.uint8)
    if qtype == "Q8_0":
        out[:, :2], out[:, 2:] = fp16(d), q.astype(np.int8).view(np.uint8)
    elif qtype == "Q2_K":
        out[:, :16] = np.asarray(scales, np.uint8) | (np.asarray(mins, np.uint8) << 4)
        out[:, 16:80] = np.bitwise_or.reduce((q.astype(np.uint8).reshape(count, 2, 4, 32) << np.arange(0, 8, 2, dtype=np.uint8)[None, None, :, None]), axis=2).reshape(count, 64)
        out[:, 80:82], out[:, 82:84] = fp16(d), fp16(dmin)
    elif qtype == "Q3_K":
        out[:, :32] = np.bitwise_or.reduce((q >= 0).astype(np.uint8).reshape(count, 8, 32) << np.arange(8, dtype=np.uint8)[None, :, None], axis=1)
        out[:, 32:96] = np.bitwise_or.reduce((q.astype(np.uint8) & 3).reshape(count, 2, 4, 32) << np.arange(0, 8, 2, dtype=np.uint8)[None, None, :, None], axis=2).reshape(count, 64)
        s = (np.asarray(scales, np.int16) + 32).astype(np.uint8)
        out[:, 96:104] = (s[:, :8] & 15) | ((s[:, 8:] & 15) << 4)
        out[:, 104:108] = np.bitwise_or.reduce((s.reshape(count, 4, 4) >> 4) << np.arange(0, 8, 2, dtype=np.uint8)[None, :, None], axis=1)
        out[:, 108:110] = fp16(d)
    elif qtype == "Q4_K":
        s, m = np.asarray(scales, np.uint8), np.asarray(mins, np.uint8)
        out[:, :2], out[:, 2:4] = fp16(d), fp16(dmin)
        out[:, 4:8] = (s[:, :4] & 63) | ((s[:, 4:] >> 4) << 6)
        out[:, 8:12] = (m[:, :4] & 63) | ((m[:, 4:] >> 4) << 6)
        out[:, 12:16] = (s[:, 4:] & 15) | ((m[:, 4:] & 15) << 4)
        q = q.astype(np.uint8).reshape(count, 4, 2, 32)
        out[:, 16:] = (q[:, :, 0] | (q[:, :, 1] << 4)).reshape(count, 128)
    elif qtype == "Q6_K":
        q = (q+32).astype(np.uint8)
        lo = (q & 15).reshape(count, 2, 2, 64)
        out[:, :128] = (lo[:, :, 0] | (lo[:, :, 1] << 4)).reshape(count, 128)
        out[:, 128:192] = np.bitwise_or.reduce((q >> 4).reshape(count, 2, 4, 32) << np.arange(0, 8, 2, dtype=np.uint8)[None, None, :, None], axis=2).reshape(count, 64)
        out[:, 192:208], out[:, 208:210] = np.asarray(scales, np.int8).view(np.uint8), fp16(d)
    else:
        raise ValueError(qtype)
    return out.reshape(-1)


def decode(raw, qtype, shape):
    if qtype in ("F16", "F32"):
        return np.asarray(raw, np.uint8).view(np.float16 if qtype == "F16" else np.float32).astype(np.float32).reshape(shape)
    q, scales, d, mins, dmin = unpack(raw, qtype)
    groups = scales.shape[1]
    values = (q.reshape(len(q), groups, -1).astype(np.float32) * (scales*d)[:, :, None])
    if mins is not None:
        values -= (mins*dmin)[:, :, None]
    return values.reshape(shape)
