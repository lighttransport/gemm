"""Independent CPU bodies for deterministic inverse-dynamics probes."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import math
import queue
import threading

import numpy as np

from .biomechanics_native import Session


def validate_workers(workers, threads):
    if (type(threads) is not int or not 1 <= threads <= 256 or type(workers) is not int
            or not 1 <= workers <= 32 or workers > threads):
        raise ValueError('inverse workers must be 1..32 and no greater than threads (1..256)')


def validate_iterations(iterations):
    if type(iterations) is not int or not 0<=iterations<=20:
        raise ValueError('inverse iterations must be 0..20')


def validate_reuse_frames(frames):
    if type(frames) is not int or not 0<=frames<=60:
        raise ValueError('sensitivity reuse frames must be 0..60')


def validate_probe_substeps(steps):
    if type(steps) is not int or not 0<=steps<=48:
        raise ValueError('sensitivity probe substeps must be 0..48')


def validate_state_precision(precision):
    if type(precision) is not str or precision not in ('auto','float32','float64'):
        raise ValueError('native state precision must be auto, float32 or float64')


def validate_mechanics_solver(solver,precision='auto'):
    if type(solver) is not str or solver not in ('partitioned','coupled','block'):
        raise ValueError('mechanics solver must be partitioned, coupled or block')
    if solver in ('coupled','block') and precision=='float32':
        raise ValueError('coupled mechanics requires auto or float64 state precision')


def response_gains(muscles,activation,excitation,deltas,dt,full_steps,probe_steps):
    """Free inertial response estimate using the native activation recurrence.

    This scales a local derivative, never a simulated state. Material/contact
    propagation is subsequently calibrated by one full native directional probe.
    """
    def response(steps):
        base=np.asarray(activation,np.float64).copy();trial=base.copy();result=np.zeros_like(base)
        on=np.array([m.activation_time_constant for m in muscles])
        off=np.array([m.deactivation_time_constant for m in muscles])
        for index in range(steps):
            base+=(excitation-base)*(-np.expm1(-dt/np.where(excitation>base,on,off)))
            probe=excitation+deltas
            trial+=(probe-trial)*(-np.expm1(-dt/np.where(probe>trial,on,off)))
            result+=(steps-index)*(trial-base)
        return result
    short=response(probe_steps);full=response(full_steps)
    return np.divide(full,short,out=np.ones_like(full),where=abs(short)>1e-18)


class _Cancellation:
    def __init__(self, *events):
        self.events = events

    def is_set(self):
        return any(event is not None and event.is_set() for event in self.events)


class SensitivityCache:
    """Optional local response model; every proposed control is forward tested."""
    def __init__(self,reuse_frames):
        if type(reuse_frames) is not int or not 1<=reuse_frames<=60:
            raise ValueError('sensitivity reuse frames must be 1..60')
        self.reuse_frames=reuse_frames
        self.clear()

    def clear(self):
        self.owner=None;self.seconds=None;self.derivatives=None;self.uses=0

    def compatible(self,state,seconds):
        return (self.owner is state and self.derivatives is not None and self.uses<self.reuse_frames
                and math.isclose(self.seconds,seconds,rel_tol=1e-10,abs_tol=1e-12))

    def refresh(self,state,seconds,columns):
        derivatives=np.asarray(columns,np.float64)
        if derivatives.shape!=(state.nm,len(state.surface_rest),3) or not np.isfinite(derivatives).all():
            raise ValueError('invalid full-surface muscle sensitivities')
        self.derivatives=derivatives.copy();self.derivatives.flags.writeable=False
        self.owner=state;self.seconds=seconds;self.uses=0

    def jacobian(self,ids):
        return self.derivatives[:,ids].reshape(len(self.derivatives),-1).T


class NativeTrialPool:
    """Persistent native sessions; the accepted trajectory stays on the owner.

    ctypes releases the GIL during native calls. Each worker owns all mutable
    native state, and results retain input order regardless of completion order.
    The shared snapshot is read-only and remains alive until all probes finish.
    """
    def __init__(self, owner, workers, *, lazy=False):
        if type(workers) is not int or not 1 <= workers <= 32:
            raise ValueError('inverse workers must be 1..32')
        self.owner = owner
        self.workers=workers;self.closed=False
        self.sessions = []
        self.available = queue.Queue()
        self.executor = None
        self.running = threading.Lock()
        if not lazy:self._start()

    def _start(self):
        owner=self.owner
        _, iterations, contacts, implicit, *_ = owner.configuration
        try:
            # Build serially: constructor failures cannot leak other bodies.
            for _ in range(self.workers):
                if owner.cancel is not None and owner.cancel.is_set():
                    raise RuntimeError('biomechanical inverse fitting cancelled')
                session = Session(owner.a, owner.metadata, path=owner.lib._name,
                                  iterations=iterations, contacts=contacts, implicit=implicit,state_precision=owner.state_precision,
                                  contact_transport=owner.contact_transport,jaw_mode=owner.jaw_mode,
                                  jaw_inertia=owner.jaw_inertia,jaw_damping=owner.jaw_damping,jaw_track=owner.jaw_track,
                                  inertial_repair_sweeps=owner.inertial_repair_sweeps,volume_rescue_limit=owner.volume_rescue_limit)
                self.sessions.append(session)
                self.available.put(session)
            self.executor = ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix='vhuman-muscle')
        except BaseException:
            self.close()
            raise

    def residuals(self, initial, probes, seconds, ids, target, *, cancel=None, progress=None):
        """Return residuals or None for unsafe trials, without mutating owner."""
        surfaces=self.surfaces(initial,probes,seconds,cancel=cancel,progress=progress)
        return [(surface[ids]-target[ids]).reshape(-1)/.001 if surface is not None else None for surface in surfaces]

    def surfaces(self, initial, probes, seconds, *, cancel=None, progress=None):
        """Return complete surfaces in probe order; None marks an unsafe trial."""
        if self.closed:
            raise RuntimeError('closed inverse trial pool')
        if initial.owner is not self.owner:
            raise ValueError('inverse trial pool belongs to a different session')
        if not self.running.acquire(blocking=False):
            raise RuntimeError('inverse trial pool is already evaluating')
        stop = threading.Event()
        interrupted = _Cancellation(stop, cancel, self.owner.cancel)
        futures = []
        results = [None] * len(probes)

        def evaluate(probe):
            if interrupted.is_set():
                raise RuntimeError('biomechanical inverse fitting cancelled')
            worker = self.available.get()
            try:
                worker.cancel = interrupted
                initial.restore_into(worker)
                try:
                    worker.advance(probe, seconds)
                except FloatingPointError:
                    return None
                return worker.surface()
            finally:
                worker.cancel = None
                self.available.put(worker)

        try:
            if self.executor is None:self._start()
            for probe in probes:
                futures.append(self.executor.submit(evaluate, probe.copy()))
            indices = {future: index for index, future in enumerate(futures)}
            for completed, future in enumerate(as_completed(futures), 1):
                results[indices[future]] = future.result()
                if progress:
                    progress(completed, len(probes))
            return results
        finally:
            # Cancellation and progress exceptions must finish native calls
            # before the caller destroys the snapshot or any worker body.
            stop.set()
            for future in futures:
                future.cancel()
            for future in futures:
                if not future.cancelled():
                    try:
                        future.result()
                    except BaseException:
                        pass
            self.running.release()

    def close(self):
        self.closed=True
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)
            self.executor = None
        for session in self.sessions:
            session.close()
        self.sessions.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
