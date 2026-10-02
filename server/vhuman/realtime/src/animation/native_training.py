"""Native two-layer GRU training. NumPy supplies data/initialization only."""
import ctypes as C
import threading
import numpy as np
from ....native_training import library, pointer, check, set_threads, IP


def parameter_shapes(hidden, controls):
    shapes = {'hidden_projection.weight': (128, hidden), 'hidden_projection.bias': (128,),
              'code_embedding.weight': (2048, 16)}
    for layer in range(2):
        shapes.update({f'recurrent.weight_ih_l{layer}': (384, 384 if layer == 0 else 128),
                       f'recurrent.weight_hh_l{layer}': (384, 128),
                       f'recurrent.bias_ih_l{layer}': (384,), f'recurrent.bias_hh_l{layer}': (384,)})
    shapes.update({'output.weight': (8*controls, 128), 'output.bias': (8*controls,)})
    return shapes


class MotionTrainer:
    def __init__(self, hidden, controls, seed=7, threads=4):
        self.handle = None
        self.lock = threading.RLock()
        if type(hidden) is not int or not 1 <= hidden <= 16384 or type(controls) is not int or not 1 <= controls <= 512:
            raise ValueError('invalid motion dimensions')
        self.hidden, self.controls = hidden, controls
        set_threads(threads)
        self.threads = threads
        self.shapes = parameter_shapes(hidden, controls)
        self.lib = library()
        self.handle = self.lib.vh_train_motion_open(hidden, controls)
        if not self.handle:
            raise RuntimeError(self.lib.vh_train_error().decode())
        count = self.lib.vh_train_motion_size(self.handle)
        if count != sum(np.prod(shape) for shape in self.shapes.values()):
            self.close()
            raise RuntimeError('native training parameter ABI mismatch')
        self._parameters = np.ctypeslib.as_array(self.lib.vh_train_motion_data(self.handle, 0), shape=(count,))
        self._gradients = np.ctypeslib.as_array(self.lib.vh_train_motion_data(self.handle, 1), shape=(count,))
        rng, offset = np.random.default_rng(seed), 0
        for name, shape in self.shapes.items():
            count = int(np.prod(shape))
            bound = 1/np.sqrt(hidden if name.startswith('hidden_projection') else 128)
            values = rng.normal(size=shape) if name == 'code_embedding.weight' else rng.uniform(-bound, bound, shape)
            self._parameters[offset:offset+count] = values.ravel()
            offset += count

    def state_dict(self):
        with self.lock:
            if not self.handle: raise RuntimeError('motion trainer is closed')
            offset, result = 0, {}
            for name, shape in self.shapes.items():
                count = int(np.prod(shape))
                result[name] = self._parameters[offset:offset+count].reshape(shape).copy()
                offset += count
            return result

    def load_state_dict(self, tensors):
        with self.lock:
            if not self.handle: raise RuntimeError('motion trainer is closed')
            if set(tensors) != set(self.shapes): raise ValueError('motion training tensor names differ')
            values = []
            for name, shape in self.shapes.items():
                value = np.asarray(tensors[name], np.float32)
                if value.shape != shape or not np.isfinite(value).all(): raise ValueError('invalid motion weights: '+name)
                values.append(value.ravel())
            self._parameters[:] = np.concatenate(values)

    def gradients(self):
        with self.lock:
            if not self.handle: raise RuntimeError('motion trainer is closed')
            offset, result = 0, {}
            for name, shape in self.shapes.items():
                count = int(np.prod(shape))
                result[name] = self._gradients[offset:offset+count].reshape(shape).copy()
                offset += count
            return result

    def compute(self, hidden, codes, target, bounds, weights, state=None, *, backward=False, update=False, lr=.001):
        hidden, target = np.ascontiguousarray(hidden, np.float32), np.ascontiguousarray(target, np.float32)
        bounds, weights = np.ascontiguousarray(bounds, np.float32), np.ascontiguousarray(weights, np.float32)
        codes = np.asarray(codes)
        t = len(hidden)
        if (hidden.shape != (t, self.hidden) or not 1 <= t <= 32 or codes.shape != (t, 16) or
                codes.dtype.kind not in 'iu' or (codes < 0).any() or (codes >= 2048).any() or
                target.shape != (t, 8, self.controls) or bounds.shape != (self.controls, 2) or
                weights.shape != (self.controls,)):
            raise ValueError('invalid motion training batch')
        codes = np.ascontiguousarray(codes, np.int32)
        state = np.zeros((2,128),np.float32) if state is None else np.array(state,dtype=np.float32,order='C',copy=True)
        if state.shape != (2,128): raise ValueError('invalid recurrent state')
        prediction = np.empty_like(target)
        loss = C.c_double()
        with self.lock:
            if not self.handle: raise RuntimeError('motion trainer is closed')
            set_threads(self.threads)
            check(self.lib.vh_train_motion_compute(self.handle, pointer(hidden), codes.ctypes.data_as(IP),
                pointer(target), pointer(bounds), pointer(weights), pointer(state), pointer(prediction),
                t, 2 if update else int(backward), lr, C.byref(loss)))
        return loss.value, prediction, state

    def close(self):
        with self.lock:
            if self.handle:
                self.lib.vh_train_motion_close(self.handle)
                self.handle = None
                self._parameters = self._gradients = None

    def __enter__(self): return self
    def __exit__(self, *unused): self.close()
    def __del__(self): self.close()
