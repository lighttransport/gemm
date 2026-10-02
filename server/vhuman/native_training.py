"""Repository CPU GEMM/AdamW and bounded PCA for framework-free training."""
import ctypes as C
from functools import lru_cache
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FP = C.POINTER(C.c_float)
IP = C.POINTER(C.c_int32)


@lru_cache(maxsize=1)
def library():
    path = ROOT/'cpu/vhuman/libvhuman_training.so'
    if not path.is_file():
        raise RuntimeError('build with make -C cpu/vhuman libvhuman_training.so')
    lib = C.CDLL(str(path))
    declarations = {
        'vh_train_error': ([], C.c_char_p),
        'vh_train_set_threads': ([C.c_int], C.c_int),
        'vh_train_gemm': ([FP, FP, FP, C.c_int, C.c_int, C.c_int, C.c_int, C.c_int], C.c_int),
        'vh_train_adamw': ([FP, FP, FP, FP, C.c_size_t, C.c_int, C.c_double, C.c_double, C.c_double], C.c_int),
        'vh_train_motion_open': ([C.c_int, C.c_int], C.c_void_p),
        'vh_train_motion_close': ([C.c_void_p], None),
        'vh_train_motion_data': ([C.c_void_p, C.c_int], FP),
        'vh_train_motion_size': ([C.c_void_p], C.c_size_t),
        'vh_train_motion_compute': ([C.c_void_p, FP, IP, FP, FP, FP, FP, FP,
                                    C.c_int, C.c_int, C.c_double, C.POINTER(C.c_double)], C.c_int),
        'vh_train_mlp': ([FP, FP, FP, FP, FP, C.c_int, C.c_int, C.c_int, C.c_int], C.c_int),
        'vh_train_spheres': ([FP, IP, FP, FP, C.c_int, C.c_int, C.c_int, C.c_int, FP, FP, FP], C.c_int),
        'vh_train_pairs': ([FP, IP, IP, FP, FP, C.c_int, C.c_int, C.c_int, FP, FP, FP], C.c_int),
        'vh_train_arap': ([FP, FP, IP, FP, C.c_int, C.c_int, C.c_int, C.c_double, FP, FP], C.c_int),
        'vh_train_rotations': ([FP, FP, FP, FP, IP, IP, C.c_int, C.c_int, C.c_int, C.c_int, FP], C.c_int),
    }
    for name, (arguments, result) in declarations.items():
        function = getattr(lib, name)
        function.argtypes, function.restype = arguments, result
    return lib


def pointer(value):
    return value.ctypes.data_as(FP)


def check(code):
    if code:
        raise ValueError(library().vh_train_error().decode())


def set_threads(threads):
    if type(threads) is not int or not 1 <= threads <= 64:
        raise ValueError('threads must be 1..64')
    check(library().vh_train_set_threads(threads))


def matmul(a, b, *, transpose_a=False, transpose_b=False):
    a, b = np.ascontiguousarray(a, np.float32), np.ascontiguousarray(b, np.float32)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError('training GEMM requires matrices')
    m, k = a.shape[::-1] if transpose_a else a.shape
    kk, n = b.shape[::-1] if transpose_b else b.shape
    if k != kk or min(m, n, k) < 1 or max(m, n, k) > 2**31-1:
        raise ValueError('invalid training GEMM dimensions')
    out = np.empty((m, n), np.float32)
    check(library().vh_train_gemm(pointer(out), pointer(a), pointer(b), m, n, k,
                                 int(transpose_a), int(transpose_b)))
    return out


class AdamW:
    """In-place optimizer; moments are ordinary NumPy storage, math is native."""
    def __init__(self, parameters, lr=.001, weight_decay=.01, clip=0):
        if parameters.dtype != np.float32 or not parameters.flags.c_contiguous or not parameters.flags.writeable:
            raise ValueError('optimizer parameters must be writable contiguous float32')
        self.parameters = parameters
        self.first, self.second = np.zeros_like(parameters), np.zeros_like(parameters)
        self.lr, self.decay, self.clip, self.iteration = lr, weight_decay, clip, 0

    def step(self, gradients, *, lr=None):
        gradients = np.ascontiguousarray(gradients, np.float32)
        if gradients.shape != self.parameters.shape:
            raise ValueError('optimizer gradient shape mismatch')
        check(library().vh_train_adamw(pointer(self.parameters), pointer(gradients), pointer(self.first),
            pointer(self.second), gradients.size, self.iteration+1, self.lr if lr is None else lr, self.decay, self.clip))
        self.iteration += 1


def randomized_basis(matrix, rank, *, seed=23, iterations=2, oversample=4):
    """Uncentered randomized SVD; thin QR/SVD use NumPy's standard LAPACK.

    All large matrix products use repository GEMM. Never form the D-by-D
    covariance or a full spatial SVD for N-by-D vertex residuals.
    """
    matrix = np.ascontiguousarray(matrix, np.float32)
    if (matrix.ndim != 2 or type(rank) is not int or not 1 <= rank <= min(matrix.shape) or
            not np.isfinite(matrix).all() or type(iterations) is not int or not 0 <= iterations <= 8 or
            type(oversample) is not int or not 0 <= oversample <= 32):
        raise ValueError('invalid bounded PCA configuration')
    q = min(rank+oversample, min(matrix.shape))
    rng = np.random.default_rng(seed)
    omega = rng.normal(size=(matrix.shape[1], q)).astype(np.float32)
    vectors = np.linalg.qr(matmul(matrix, omega), mode='reduced')[0]
    for _ in range(iterations):
        spatial = np.linalg.qr(matmul(matrix, vectors, transpose_a=True), mode='reduced')[0]
        vectors = np.linalg.qr(matmul(matrix, spatial), mode='reduced')[0]
    small = matmul(vectors, matrix, transpose_a=True)
    _, _, basis = np.linalg.svd(small, full_matrices=False)
    return np.ascontiguousarray(basis[:rank], np.float32)
