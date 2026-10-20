"""CPU-only LightGeom active tissue, surface contact and mandible bindings.

All dynamic state, including excitation dynamics and jaw reactions, is part
of Snapshot. Inverse-control trials never advance the accepted trajectory.
"""
from __future__ import annotations

import ctypes as ct
import weakref
import hashlib
import math
import os
import uuid
from pathlib import Path

import numpy as np
from .jaw_model import DEFAULT_TRACK

ROOT = Path(__file__).resolve().parents[3]
LIGHTGEOM = ROOT / 'third_party/LightGeom'
LIBRARY = ROOT / 'tmp/vhuman-full-native/examples/lightphysics_vhuman_face/liblightphysics_vhuman_full.so'
NATIVE_VOLUME_FLOOR = .06  # Headroom for legacy float32 transport and export.
P = ct.c_void_p
I = ct.c_int32
D = ct.c_double
F = ct.c_float
CANCEL_CALLBACK=ct.CFUNCTYPE(I,P)


def ptr(array):
    return array.ctypes.data_as(P) if array is not None else None


def freeze_library(path,cache=None):
    """Publish immutable solver bytes so rebuilds cannot change a live bake."""
    data=Path(path).read_bytes();digest=hashlib.sha256(data).hexdigest()
    cache=Path(cache) if cache is not None else ROOT/'tmp/vhuman-native-solvers'
    cache.mkdir(parents=True,exist_ok=True);destination=cache/(digest+'.so')
    if not destination.exists():
        temporary=cache/('.'+digest+'-'+uuid.uuid4().hex)
        try:
            with temporary.open('xb') as stream:
                stream.write(data);stream.flush();os.fsync(stream.fileno())
            temporary.chmod(0o555)
            try:os.link(temporary,destination)
            except FileExistsError:pass
        finally:temporary.unlink(missing_ok=True)
    if hashlib.sha256(destination.read_bytes()).hexdigest()!=digest:
        raise ValueError('frozen CPU solver hash mismatch')
    return destination,digest


class Descriptor(ct.Structure):
    _fields_ = [('nverts', I), ('rest_positions', P), ('initial_velocities', P), ('inverse_mass', P),
                ('nedges', I), ('edges', P), ('ntets', I), ('tets', P), ('nfaces', I), ('faces', P)]


class Options(ct.Structure):
    _fields_ = [('substeps', I), ('iterations', I), ('distance_compliance', F), ('volume_compliance', F),
                ('attachment_compliance', F), ('damping', F), ('gravity', F * 3), ('self_collision', I),
                ('self_collision_thickness', F), ('self_collision_compliance', F), ('friction', F),
                ('restitution', F), ('contact_compliance', F), ('bending_compliance', F), ('pressure', F),
                ('plasticity', F), ('plasticity_threshold', F), ('plasticity_max', F), ('min_volume_ratio', F),
                ('material', I), ('youngs_modulus', F), ('poisson_ratio', F)]


class Fiber(ct.Structure):
    _fields_ = [('tetrahedron', I), ('muscle', I), ('direction', D * 3), ('peak_stress', D)]


class ActiveCable(ct.Structure):
    _fields_=[('muscle',I),('vertex_count',I),('vertices',I*16),('weights',D*16),
              ('origin',D*3),('max_force',D),('rest_length',D),('shortening',D)]


def pack_cables(values,nvertices,nmuscles,*,inverse_mass=None,rest=None):
    from .tongue_cables import validate_rows
    validate_rows(values,nvertices,nmuscles,inverse_mass=inverse_mass,rest=rest)
    rows=np.asarray(values,np.float64);result=(ActiveCable*len(rows))()
    for cable,row in zip(result,rows):
        cable.muscle=int(row[0]);cable.vertex_count=int(row[1]);count=cable.vertex_count
        cable.origin[:]=row[2:5];cable.max_force,cable.rest_length,cable.shortening=row[5:8]
        cable.vertices[:count]=row[8:8+count].astype(np.int32);cable.weights[:count]=row[24:24+count]
    return result


class Muscle(ct.Structure):
    _fields_ = [('activation', D), ('activation_time_constant', D), ('deactivation_time_constant', D)]


class JawMuscle(ct.Structure):
    _fields_ = [('muscle', I), ('origin', D * 3), ('insertion', D * 3), ('max_force', D), ('optimal_length', D)]


class Attachment(ct.Structure):
    _fields_ = [('vertex', I), ('target', F * 3), ('weight', F), ('compliance', F)]


def pack_fibers(values):
    """Write the native structure layout directly without per-row objects."""
    values=np.asarray(values,np.float64)
    if (values.ndim!=2 or values.shape[1]!=6 or not np.isfinite(values).all()
            or (values[:,:2]<0).any() or (values[:,:2]>np.iinfo(np.int32).max).any()
            or (values[:,:2]!=np.floor(values[:,:2])).any()):
        raise ValueError('invalid native muscle fiber array')
    result=(Fiber*len(values))()
    view=np.ctypeslib.as_array(result)
    view['tetrahedron']=values[:,0];view['muscle']=values[:,1]
    view['direction']=values[:,2:5];view['peak_stress']=values[:,5]
    return result


class ContactStats(ct.Structure):
    _fields_ = [('candidates', ct.c_uint64), ('contacts', ct.c_uint64), ('max_correction', D)]


class GeometryStepError(FloatingPointError):
    """A rejected physical step with bounded, JSON-safe failure evidence."""
    def __init__(self,message,diagnostics):
        super().__init__(message)
        self.diagnostics=diagnostics


def library(path=LIBRARY):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Build LightGeom target lightphysics_vhuman_full: {path}')
    lib = ct.CDLL(str(path.resolve()))
    signatures = {
        'vhface_full_abi': (ct.c_uint32, []),
        'lightphysics_deformable_options_default': (None, [ct.POINTER(Options)]),
        'lightphysics_deformable_create': (I, [ct.POINTER(Descriptor), ct.POINTER(Options), ct.POINTER(P)]),
        'lightphysics_deformable_destroy': (None, [P]),
        'lightphysics_deformable_set_youngs_moduli': (I, [P, P, I]),
        'lightphysics_deformable_set_attachments': (I, [P, P, I]),
        'lightphysics_deformable_update_attachment_targets': (I, [P, P, I]),
        'lightphysics_deformable_read_attachment_impulses': (I, [P, P, ct.c_size_t]),
        'lightphysics_deformable_step': (I, [P, P, D, P]),
        'lightphysics_deformable_read_state': (I, [P, P, P, ct.c_size_t]),
        'lightphysics_deformable_write_state': (I, [P, P, P, ct.c_size_t]),
        'lightphysics_deformable_snapshot_save': (I, [P, ct.POINTER(P)]),
        'lightphysics_deformable_snapshot_restore': (I, [P, P]),
        'lightphysics_deformable_snapshot_destroy': (None, [P]),
        'lightphysics_deformable_enable_guarded_solve': (I, [P,I]),
        'lightphysics_deformable_project_guarded': (I, [P,P,P,I,P]),
        'lightphysics_active_tissue_create': (I, [P, P, I, P, I, P, I, I, ct.POINTER(P)]),
        'lightphysics_active_tissue_enable_implicit': (I,[P,I]),
        'lightphysics_active_tissue_destroy': (None, [P]),
        'lightphysics_active_tissue_apply': (I, [P, P, I, D]),
        'lightphysics_kinematics_muscle_network_step': (I, [P, I, P, D, P]),
        'lightphysics_surface_contact_create': (I, [P, P, I, P, I, ct.POINTER(P)]),
        'lightphysics_surface_contact_destroy': (None, [P]),
        'lightphysics_surface_contact_project': (I, [P, P, P, D, D, I, P, ct.POINTER(ContactStats)]),
        'lightphysics_surface_contact_count_intersections':(I,[P,P,ct.POINTER(ct.c_uint64)]),
        'vhface_jaw_transform': (I, [P, I, P, P, P]),
        'vhface_jaw_step': (I, [P, P, P, P, I, P, I, P, D]),
    }
    for name, (result, arguments) in signatures.items():
        function = getattr(lib, name)
        function.restype, function.argtypes = result, arguments
    find = getattr(lib, 'lightphysics_surface_contact_find_intersections', None)
    if find is not None:
        find.restype, find.argtypes = I, [P,P,P,ct.c_size_t,ct.POINTER(ct.c_uint64)]
    frames = getattr(lib,'lightphysics_deformable_set_attachment_frames',None)
    if frames is not None:
        frames.restype,frames.argtypes=I,[P,P,P,I]
    batch=getattr(lib,'lightphysics_deformable_project_guarded_batch',None)
    if batch is not None:batch.restype,batch.argtypes=I,[P,P,P,I,P]
    for name in ('read_state_f64','write_state_f64'):
        function=getattr(lib,'lightphysics_deformable_'+name,None)
        if function is not None:function.restype,function.argtypes=I,[P,P,P,ct.c_size_t]
    for name,arguments in (('set_cables',[P,P,I]),('read_cable_reactions',[P,P,ct.c_size_t])):
        function=getattr(lib,'lightphysics_active_tissue_'+name,None)
        if function is not None:function.restype,function.argtypes=I,arguments
    indexed=getattr(lib,'lightphysics_deformable_read_positions_indexed_f64',None)
    if indexed is not None:indexed.restype,indexed.argtypes=I,[P,P,I,P,ct.c_size_t]
    fractions=getattr(lib,'lightphysics_deformable_read_inertial_guard_fractions',None)
    if fractions is not None:fractions.restype,fractions.argtypes=I,[P,P,I,P,I]
    repair=getattr(lib,'lightphysics_deformable_set_inertial_repair_sweeps',None)
    if repair is not None:repair.restype,repair.argtypes=I,[P,I]
    rescue=getattr(lib,'lightphysics_deformable_set_volume_rescue_limit',None)
    if rescue is not None:rescue.restype,rescue.argtypes=I,[P,I]
    for name in ('embedded','embedded_volume'):
        embedded=getattr(lib,'lightphysics_surface_contact_project_'+name,None)
        if embedded is not None:embedded.restype,embedded.argtypes=I,[P,P,P,P,P,P,D,D,I,P,P,P]
    optional_jaw={
        'step_configured':(I,[P,P,P,P,I,P,I,P,D,P,P]),
        'step_model':(I,[P,P,P,P,I,P,I,P,D,P,P,P]),
        'transform_model':(I,[P,I,P,P,P,P]),
        'coupling_create':(I,[P,P,P,P,P,I,P,P,I,P]),
        'coupling_create_model':(I,[P,P,P,P,P,P,I,P,P,I,P]),
        'coupling_prepare':(I,[P,P,P,D]),
        'coupling_read':(I,[P,P]),
        'coupling_destroy':(None,[P]),
        'coupling_enable_volume_solve':(I,[P,I]),
        'coupling_enable_block_solve':(I,[P,I]),
        'contact_project':(I,[P,P,P,P,P,P,D,D,I,P,P,P]),
    }
    for name,(result,arguments) in optional_jaw.items():
        function=getattr(lib,'vhface_jaw_'+name,None)
        if function is not None:function.restype,function.argtypes=result,arguments
    cancel=getattr(lib,'lightphysics_deformable_set_cancel_callback',None)
    if cancel is not None:cancel.restype,cancel.argtypes=I,[P,P,P]
    if lib.vhface_full_abi() != 1:
        raise ValueError('unsupported full-face native ABI')
    return lib


def check(status):
    if status==-8:raise RuntimeError('biomechanical simulation cancelled')
    if status:
        raise RuntimeError(f'LightGeom biomechanics returned status {status}')


def volumes(positions, tetrahedra):
    p = np.asarray(positions)[tetrahedra]
    return np.einsum('ij,ij->i', np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), p[:, 3] - p[:, 0]) / 6


class Snapshot:
    def __init__(self, session):
        self.owner = session
        self.handle = P()
        check(session.lib.lightphysics_deformable_snapshot_save(session.body, ct.byref(self.handle)))
        self.muscles = bytes(session.muscles)
        self.q, self.speed, self.reaction = (a.copy() for a in (session.q, session.speed, session.reaction))
        self.positions = session.positions.copy()
        self.rigid_targets=session.rigid_targets.copy()
        self.anchor_targets=session.anchor_targets.copy()

    def restore(self):
        self._restore(self.owner)

    def restore_into(self, session):
        """Clone dynamics onto a session sharing this immutable package/config."""
        if (session.a is not self.owner.a or session.metadata is not self.owner.metadata
                or session.configuration != self.owner.configuration):
            raise ValueError('snapshot clone requires the same static anatomy and solver configuration')
        self._restore(session)

    def _restore(self, s):
        if not self.handle or not s.body:
            raise RuntimeError('closed biomechanics snapshot or session')
        check(s.lib.lightphysics_deformable_update_attachment_targets(s.body,ptr(self.anchor_targets),len(self.anchor_targets)))
        check(s.lib.lightphysics_deformable_snapshot_restore(s.body, self.handle))
        ct.memmove(s.muscles, self.muscles, len(self.muscles))
        s.activation[:] = [m.activation for m in s.muscles]
        if s.implicit:
            # Implicit apply copies activation only: it neither integrates
            # time nor injects impulses. Keep the native projector's private
            # drive synchronized even when a trial is rejected/cancelled.
            check(s.lib.lightphysics_active_tissue_apply(s.field,ptr(s.activation),s.nm,1/240))
        s.q[:], s.speed[:], s.reaction[:], s.positions[:] = self.q, self.speed, self.reaction, self.positions
        s.rigid_targets[:]=self.rigid_targets
        s.anchor_targets[:]=self.anchor_targets
        if s.coupling:
            # Synchronize private reduced coordinates without integrating time.
            # The next step prepares its own effective inertia and duration.
            inverse=np.ascontiguousarray(1/(s.jaw_inertia+s.jaw_damping/240))
            check(s.lib.vhface_jaw_coupling_prepare(s.coupling,ptr(s.q),ptr(inverse),1/240))

    def close(self):
        if self.handle:
            self.owner.lib.lightphysics_deformable_snapshot_destroy(self.handle)
            self.handle = P()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class Session:
    @property
    def cancel(self):return self._cancel

    @cancel.setter
    def cancel(self,event):
        self._cancel=event;self._cancel_error=None
        setter=getattr(self.lib,'lightphysics_deformable_set_cancel_callback',None)
        if setter is not None and getattr(self,'body',None):
            callback=None
            if event is not None:
                if not hasattr(self,'_native_cancel'):
                    owner=weakref.ref(self)
                    def poll(_):
                        state=owner()
                        if state is None:return 1
                        try:return int(state._cancel is not None and state._cancel.is_set())
                        except BaseException as exc:state._cancel_error=exc;return 1
                    self._native_cancel=CANCEL_CALLBACK(poll)
                callback=ct.cast(self._native_cancel,P)
            check(setter(self.body,callback,None))

    def __init__(self, arrays, metadata, *, path=LIBRARY, iterations=24, contacts=True, implicit=True, cancel=None,state_precision='auto',contact_transport='proxy',
                 jaw_mode='partitioned',jaw_inertia=(.0004,.2,.2),jaw_damping=(1.,80.,80.),jaw_track=None,inertial_repair_sweeps=0,volume_rescue_limit=0):
        self.lib = library(path)
        if type(inertial_repair_sweeps) is not int or not 0<=inertial_repair_sweeps<=32:
            raise ValueError('inertial repair sweeps must be 0..32')
        self.inertial_repair_sweeps=inertial_repair_sweeps
        if inertial_repair_sweeps and getattr(self.lib,'lightphysics_deformable_set_inertial_repair_sweeps',None) is None:
            raise RuntimeError('inertial repair requires rebuilding the CPU library')
        if type(volume_rescue_limit) is not int or not 0<=volume_rescue_limit<=4096:
            raise ValueError('volume rescue limit must be 0..4096')
        self.volume_rescue_limit=volume_rescue_limit
        if volume_rescue_limit and getattr(self.lib,'lightphysics_deformable_set_volume_rescue_limit',None) is None:
            raise RuntimeError('volume rescue requires rebuilding the CPU library')
        if contacts and getattr(self.lib,'lightphysics_deformable_project_guarded_batch',None) is None:
            raise RuntimeError('surface-contact volume guards require rebuilding the CPU biomechanics library')
        self.implicit = bool(implicit)
        self.cancel = cancel
        if state_precision not in ('auto','float32','float64'):raise ValueError('invalid native state precision')
        double_state=all(getattr(self.lib,'lightphysics_deformable_'+name+'_f64',None) is not None for name in ('read_state','write_state'))
        if state_precision=='float64' and not double_state:raise RuntimeError('double state transport requires rebuilding the CPU biomechanics library')
        self.state_precision=('float64' if double_state else 'float32') if state_precision=='auto' else state_precision
        if contact_transport not in ('proxy','embedded','embedded_volume','embedded_coupled'):raise ValueError('invalid surface contact transport')
        contact_function='vhface_jaw_contact_project' if contact_transport=='embedded_coupled' else 'lightphysics_surface_contact_project_'+contact_transport
        self.project_embedded=getattr(self.lib,contact_function,None)
        if contact_transport!='proxy' and (self.state_precision!='float64' or self.project_embedded is None):
            raise RuntimeError('embedded contacts require double state and the updated CPU contact library')
        self.contact_transport=contact_transport
        if jaw_mode not in ('partitioned','coupled','coupled_volume','coupled_block'):raise ValueError('invalid jaw attachment mode')
        if contact_transport=='embedded_coupled' and jaw_mode=='partitioned':raise ValueError('coupled contacts require coupled jaw attachments')
        self.jaw_mode=jaw_mode
        self.jaw_inertia=np.array(jaw_inertia,dtype=np.float64,copy=True,order='C')
        self.jaw_damping=np.array(jaw_damping,dtype=np.float64,copy=True,order='C')
        if (self.jaw_inertia.shape!=(3,) or self.jaw_damping.shape!=(3,) or
                not np.isfinite(self.jaw_inertia).all() or not np.isfinite(self.jaw_damping).all() or
                (self.jaw_inertia<=0).any() or (self.jaw_damping<0).any()):
            raise ValueError('jaw inertia must be positive and damping nonnegative finite triples')
        self.jaw_inertia.setflags(write=False);self.jaw_damping.setflags(write=False)
        if (jaw_mode!='partitioned' and (self.state_precision!='float64' or
                not all(getattr(self.lib,'vhface_jaw_coupling_'+name,None) is not None for name in ('create','prepare','read','destroy')))):
            raise RuntimeError('coupled jaw attachments require double state and the updated CPU library')
        if jaw_mode=='coupled_volume' and getattr(self.lib,'vhface_jaw_coupling_enable_volume_solve',None) is None:
            raise RuntimeError('volume-coupled jaw attachments require the updated CPU library')
        if jaw_mode=='coupled_block' and getattr(self.lib,'vhface_jaw_coupling_enable_block_solve',None) is None:
            raise RuntimeError('block jaw attachments require the updated CPU library')
        self.configured_jaw=getattr(self.lib,'vhface_jaw_step_configured',None)
        if self.configured_jaw is None and (tuple(self.jaw_inertia)!=(.0004,.2,.2) or tuple(self.jaw_damping)!=(1.,80.,80.)):
            raise RuntimeError('configured jaw resistance requires rebuilding the CPU library')
        self.jaw_track=np.array(arrays.get('jaw_track',DEFAULT_TRACK) if jaw_track is None else jaw_track,dtype=np.float64,copy=True,order='C')
        if self.jaw_track.shape!=(3,) or not np.isfinite(self.jaw_track).all():raise ValueError('invalid jaw condylar track')
        self.jaw_track.setflags(write=False)
        self.custom_jaw_track=not np.array_equal(self.jaw_track,DEFAULT_TRACK)
        if self.custom_jaw_track and (not all(getattr(self.lib,'vhface_jaw_'+name,None) is not None for name in ('transform_model','step_model')) or
                (jaw_mode!='partitioned' and getattr(self.lib,'vhface_jaw_coupling_create_model',None) is None)):
            raise RuntimeError('fitted jaw track requires rebuilding the CPU library')
        self.state_dtype=np.float64 if self.state_precision=='float64' else np.float32
        suffix='_f64' if self.state_precision=='float64' else ''
        self.read_state=getattr(self.lib,'lightphysics_deformable_read_state'+suffix)
        self.write_state=getattr(self.lib,'lightphysics_deformable_write_state'+suffix)
        self.configuration = (self.lib._handle,int(iterations),bool(contacts),bool(implicit),
                              len(arrays['rest']),len(arrays['tets']),len(arrays['anchor_ids']),len(metadata['muscles']),self.state_precision,self.contact_transport,
                              self.jaw_mode,tuple(self.jaw_inertia),tuple(self.jaw_damping),tuple(self.jaw_track),self.inertial_repair_sweeps,self.volume_rescue_limit,
                              len(arrays.get('tongue_cables',())))
        self.cable_count=len(arrays.get('tongue_cables',()))
        self.body, self.field, self.contact, self.coupling = P(), P(), P(), P()
        self.a, self.metadata = arrays, metadata
        self.rest = np.ascontiguousarray(arrays['rest'], np.float64)
        self.positions = self.rest.copy()
        self.rigid_targets=self.rest.copy()
        self.tets = np.ascontiguousarray(arrays['tets'], np.int32)
        self.rest_volume = volumes(self.rest, self.tets)
        self.faces = np.ascontiguousarray(arrays['faces'], np.int32)
        self.mass = np.ascontiguousarray(arrays['inverse_mass'], np.float64)
        self.embed_ids = np.asarray(arrays.get('embed_ids', np.repeat(np.arange(len(self.rest))[:, None], 4, axis=1)), np.int32)
        self.embed_weights = np.asarray(arrays.get('embed_weights', np.tile([1., 0, 0, 0], (len(self.rest), 1))), np.float64)
        self.surface_rest = np.ascontiguousarray(arrays.get('surface_rest', self.rest), np.float64)
        self.surface_mass = np.ascontiguousarray((self.mass[self.embed_ids] * self.embed_weights**2).sum(1))
        if 'source_ids' in arrays:
            _,first,inverse=np.unique(arrays['source_ids'],return_index=True,return_inverse=True)
            self.contact_faces=np.ascontiguousarray(first[inverse][self.faces],np.int32)
        else:
            self.contact_faces=self.faces
        self.bone = np.asarray(arrays['bone'], np.int32)
        self.hinge = np.ascontiguousarray(arrays['jaw_hinge'], np.float64)
        self.nm = len(metadata['muscles'])
        self.activation = np.zeros(self.nm, np.float64)
        # Work counters deliberately survive physical snapshot restoration.
        self.native_steps=0;self.subdivision_retries=0
        self.q, self.speed, self.reaction = (np.zeros(3, np.float64) for _ in range(3))
        self.muscles = (Muscle * self.nm)(*[Muscle(0, .025, .050) for _ in range(self.nm)])
        fields = arrays['fibers']
        packed_fields = pack_fibers(fields)
        jaw = arrays['jaw_muscles']
        self.jaw = (JawMuscle * len(jaw))(*[JawMuscle(int(v[0]), (D * 3)(*v[1:4]), (D * 3)(*v[4:7]), float(v[7]), float(v[8])) for v in jaw])
        selected=np.asarray(arrays['anchor_bone'])!=1 if jaw_mode!='partitioned' else np.ones(len(arrays['anchor_ids']),bool)
        self.jaw_anchor_ids=np.asarray(arrays['anchor_ids'],np.int32)[np.asarray(arrays['anchor_bone'])==1]
        self.anchor_ids = np.ascontiguousarray(np.asarray(arrays['anchor_ids'])[selected], np.int32)
        self.anchor_bone = np.ascontiguousarray(np.asarray(arrays['anchor_bone'])[selected], np.int32)
        self.anchor_rest = np.ascontiguousarray(self.rest[self.anchor_ids])
        self.anchor_targets=np.ascontiguousarray(self.anchor_rest,np.float32)
        self.attachments = (Attachment * len(self.anchor_ids))(*[
            Attachment(int(v), (F * 3)(*self.rest[v]), 1, 1e-7) for v in self.anchor_ids])
        rest32 = self.rest.astype(np.float32)
        mass32 = self.mass.astype(np.float32)
        desc = Descriptor(len(self.rest), ptr(rest32), None, ptr(mass32), 0, None,
                          len(self.tets), ptr(self.tets), 0, None)
        options = Options()
        self.lib.lightphysics_deformable_options_default(ct.byref(options))
        options.substeps, options.iterations = 1, iterations
        options.material, options.youngs_modulus, options.poisson_ratio = 2, 10000, .45
        options.volume_compliance, options.damping, options.min_volume_ratio = 0, 1, NATIVE_VOLUME_FLOOR
        options.gravity[:] = (0, 0, 0)  # Neutral-equilibrium head-local model.
        try:
            check(self.lib.lightphysics_deformable_create(ct.byref(desc), ct.byref(options), ct.byref(self.body)))
            if self.inertial_repair_sweeps:
                check(self.lib.lightphysics_deformable_set_inertial_repair_sweeps(self.body,self.inertial_repair_sweeps))
            if self.volume_rescue_limit:
                check(self.lib.lightphysics_deformable_set_volume_rescue_limit(self.body,self.volume_rescue_limit))
            self.cancel=cancel
            moduli = np.ascontiguousarray(arrays['youngs_moduli'], np.float32)
            check(self.lib.lightphysics_deformable_set_youngs_moduli(self.body, ptr(moduli), len(moduli)))
            check(self.lib.lightphysics_deformable_set_attachments(self.body, self.attachments, len(self.attachments)))
            if 'anchor_normals' in arrays:
                frames=getattr(self.lib,'lightphysics_deformable_set_attachment_frames',None)
                if frames is None:
                    raise RuntimeError('sliding skin attachments require rebuilding the CPU biomechanics library')
                normals=np.ascontiguousarray(arrays['anchor_normals'][selected],np.float64)
                tangent=np.ascontiguousarray(arrays['anchor_tangent_compliance'][selected],np.float64)
                check(frames(self.body,ptr(normals),ptr(tangent),len(self.attachments)))
            check(self.lib.lightphysics_active_tissue_create(self.body, ptr(rest32), len(self.rest), ptr(self.tets),
                  len(self.tets), packed_fields, len(packed_fields), self.nm, ct.byref(self.field)))
            # The native field copies this ABI staging buffer at creation.
            del packed_fields
            if self.cable_count:
                if any(getattr(self.lib,'lightphysics_active_tissue_'+name,None) is None for name in ('set_cables','read_cable_reactions')):
                    raise RuntimeError('external muscle cables require rebuilding the CPU library')
                cables=pack_cables(arrays['tongue_cables'],len(self.rest),self.nm,inverse_mass=self.mass,rest=self.rest)
                check(self.lib.lightphysics_active_tissue_set_cables(self.field,cables,len(cables)))
            check(self.lib.lightphysics_active_tissue_enable_implicit(self.field,int(implicit)))
            if self.jaw_mode!='partitioned':
                rigid_ids=np.flatnonzero(self.bone==1).astype(np.int32)
                moving_rest=np.ascontiguousarray(self.rest[self.jaw_anchor_ids])
                rigid_rest=np.ascontiguousarray(self.rest[rigid_ids])
                compliance=np.full(len(self.jaw_anchor_ids),1e-7)
                arguments=(ptr(self.jaw_anchor_ids),ptr(moving_rest),ptr(compliance),len(self.jaw_anchor_ids),
                           ptr(rigid_ids),ptr(rigid_rest),len(rigid_ids),ct.byref(self.coupling))
                if self.custom_jaw_track:
                    check(self.lib.vhface_jaw_coupling_create_model(self.body,ptr(self.hinge),ptr(self.jaw_track),*arguments))
                else:check(self.lib.vhface_jaw_coupling_create(self.body,ptr(self.hinge),*arguments))
                if self.jaw_mode=='coupled_volume':check(self.lib.vhface_jaw_coupling_enable_volume_solve(self.coupling,1))
                if self.jaw_mode=='coupled_block':check(self.lib.vhface_jaw_coupling_enable_block_solve(self.coupling,1))
            if contacts:
                check(self.lib.lightphysics_surface_contact_create(ptr(self.surface_rest), ptr(self.surface_mass), len(self.surface_rest),
                      ptr(self.contact_faces), len(self.contact_faces), ct.byref(self.contact)))
        except BaseException:
            self.close()
            raise

    def close(self):
        if getattr(self,'coupling',None):
            self.lib.vhface_jaw_coupling_destroy(self.coupling);self.coupling=P()
        for handle, name in (('contact', 'surface_contact'), ('field', 'active_tissue'), ('body', 'deformable')):
            if getattr(self, handle, None):
                getattr(self.lib, f'lightphysics_{name}_destroy')(getattr(self, handle))
                setattr(self, handle, P())
        # Release the callback wrapper after native destruction. Its closure
        # uses a weak owner so closed arrays need not await cyclic collection.
        self._cancel=None;self._cancel_error=None
        if hasattr(self,'_native_cancel'):del self._native_cancel

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def transform(self, points):
        points = np.ascontiguousarray(points, np.float64)
        result = np.empty_like(points)
        if self.custom_jaw_track:
            check(self.lib.vhface_jaw_transform_model(ptr(points),len(points),ptr(self.hinge),ptr(self.q),ptr(self.jaw_track),ptr(result)))
        else:check(self.lib.vhface_jaw_transform(ptr(points), len(points), ptr(self.hinge), ptr(self.q), ptr(result)))
        return result

    def surface(self, positions=None):
        p = self.positions if positions is None else positions
        return (p[self.embed_ids] * self.embed_weights[..., None]).sum(1)

    def cable_reactions(self):
        """Last substep's paired fixed-origin impulses, in N s."""
        result=np.zeros((self.cable_count,3),np.float64)
        if self.cable_count:
            check(self.lib.lightphysics_active_tissue_read_cable_reactions(self.field,ptr(result),result.size))
        return result

    def intersections(self, surface=None):
        if not self.contact:
            raise RuntimeError('surface contact checking disabled')
        count=ct.c_uint64()
        surface=np.ascontiguousarray(self.surface() if surface is None else surface,np.float64)
        if surface.shape!=self.surface_rest.shape or not np.isfinite(surface).all():
            raise ValueError('invalid surface crossing query')
        check(self.lib.lightphysics_surface_contact_count_intersections(self.contact,ptr(surface),ct.byref(count)))
        return int(count.value)

    def generalized(self, points, forces):
        relative = points - self.hinge
        if self.custom_jaw_track:
            relative=relative-self.q[0]*self.jaw_track
            angle=np.column_stack((np.full(len(points),self.jaw_track[0]),
                -relative[:,2]+self.q[2]+self.jaw_track[1],relative[:,1]+self.jaw_track[2]))
        else:
            angle = np.column_stack((np.zeros(len(points)), -relative[:, 2] + self.q[2] - .003333333333333333,
                                     relative[:, 1] + .003333333333333333 * self.q[0]))
        return np.array([(angle * forces).sum(), forces[:, 0].sum(), forces[:, 2].sum()])

    def failure_geometry(self,positions,dt,*,surface_correction=None,contact_fraction=None):
        """Capture the failed trial before transactional rollback removes it."""
        ratio=volumes(positions,self.tets)/self.rest_volume
        ids=np.argsort(np.nan_to_num(ratio,nan=-np.inf),kind='stable')[:16]
        result=dict(dt_seconds=float(dt),minimum_tetrahedra=[
            dict(tetrahedron=int(i),vertices=self.tets[i].tolist(),
                 volume_ratio=float(ratio[i]) if np.isfinite(ratio[i]) else None) for i in ids])
        if contact_fraction is not None:result['contact_volume_fraction_min']=float(contact_fraction)
        if surface_correction is not None:
            distance=np.linalg.norm(surface_correction,axis=1)*1000
            ids=np.argsort(-distance,kind='stable')[:16]
            result['contact_vertices']=[]
            for i in ids:
                if not distance[i]>0:continue
                part=next((p['name'] for p in self.metadata['parts']
                           if p['surface_start']<=i<p['surface_start']+p['surface_count']),None)
                result['contact_vertices'].append(dict(surface_vertex=int(i),source_vertex=int(self.a['source_ids'][i]),
                    part=part,remaining_correction_mm=float(distance[i]),
                    embedded_vertices=self.embed_ids[i].tolist()))
        return result

    def _step(self, excitation, dt):
        if not self.body:
            raise RuntimeError('closed biomechanics session')
        excitation = np.ascontiguousarray(excitation, np.float64)
        if excitation.shape != (self.nm,) or not np.isfinite(excitation).all() or (excitation < 0).any() or (excitation > 1).any():
            raise ValueError('muscle excitations must be finite and in [0,1]')
        check(self.lib.lightphysics_kinematics_muscle_network_step(self.muscles, self.nm, ptr(excitation), dt, ptr(self.activation)))
        old_q=self.q.copy()
        arguments=(ptr(self.q),ptr(self.speed),ptr(self.hinge),self.jaw,len(self.jaw),ptr(self.activation),self.nm,ptr(self.reaction),dt)
        if self.custom_jaw_track:
            check(self.lib.vhface_jaw_step_model(*arguments,ptr(self.jaw_inertia),ptr(self.jaw_damping),ptr(self.jaw_track)))
        elif self.configured_jaw is None:check(self.lib.vhface_jaw_step(*arguments))
        else:check(self.configured_jaw(*arguments,ptr(self.jaw_inertia),ptr(self.jaw_damping)))
        if self.coupling:
            inverse=np.ascontiguousarray(1/(self.jaw_inertia+dt*self.jaw_damping))
            check(self.lib.vhface_jaw_coupling_prepare(self.coupling,ptr(self.q),ptr(inverse),dt))
        x, velocity = (np.empty(self.rest.shape, self.state_dtype) for _ in range(2))
        check(self.read_state(self.body, ptr(x), ptr(velocity), x.size))
        previous = x.astype(np.float64)
        fixed=(self.bone==0)&(self.mass==0)
        x[fixed]=self.rigid_targets[fixed]
        rigid = self.bone == 1
        x[rigid] = self.transform(self.rest[rigid])
        check(self.write_state(self.body, ptr(x), ptr(velocity), x.size))
        targets = self.anchor_rest.copy()
        moving = self.anchor_bone == 1
        targets[moving] = self.transform(targets[moving])
        targets = np.ascontiguousarray(targets, np.float32)
        self.anchor_targets[:]=targets
        check(self.lib.lightphysics_deformable_update_attachment_targets(self.body, ptr(targets), len(targets)))
        check(self.lib.lightphysics_active_tissue_apply(self.field, ptr(self.activation), self.nm, dt))
        self.native_steps+=1
        output=x if self.state_precision=='float32' else np.empty(self.rest.shape,np.float32)
        status=self.lib.lightphysics_deformable_step(self.body, None, dt, ptr(output))
        if self._cancel_error is not None:raise self._cancel_error
        check(status)
        if self.coupling:
            check(self.lib.vhface_jaw_coupling_read(self.coupling,ptr(self.q)))
            self.speed[:]=(self.q-old_q)/dt
        if self.state_precision=='float64':check(self.read_state(self.body,ptr(x),None,x.size))
        if not np.isfinite(x).all():
            raise FloatingPointError('full-face constitutive solve produced nonfinite positions')
        impulses = np.zeros((len(self.anchor_ids), 3), np.float64)
        check(self.lib.lightphysics_deformable_read_attachment_impulses(self.body, ptr(impulses), impulses.size))
        # XPBD lambda has units kg m. Convert its opposite reaction to force.
        self.reaction[:] = self.generalized(targets[moving], -impulses[moving] / dt**2)
        if not np.isfinite(self.reaction).all():
            raise FloatingPointError('full-face attachment solve produced nonfinite jaw reaction')
        result = x.astype(np.float64)
        stats = ContactStats()
        residual=0.
        contact_fraction=1.
        actual_contact_correction=0.
        if self.contact:
            before = np.ascontiguousarray(self.surface(previous))
            adaptive=self.contact_transport in ('embedded_volume','embedded_coupled')
            local_crossings=export_crossings=0
            for contact_pass in range(12 if adaptive else 3):
                if self.cancel is not None and self.cancel.is_set():raise RuntimeError('biomechanical simulation cancelled')
                surface = np.ascontiguousarray(self.surface(result))
                corrected = surface.copy()
                reaction = np.zeros_like(surface)
                current = ContactStats()
                if self.contact_transport!='proxy':
                    fraction=D(1.)
                    contact_owner=(self.coupling,self.contact) if self.contact_transport=='embedded_coupled' else (self.contact,self.body)
                    check(self.project_embedded(
                        *contact_owner,ptr(self.embed_ids),ptr(self.embed_weights),ptr(before),ptr(corrected),
                        dt,.0002,1,ptr(reaction),ct.byref(current),ct.byref(fraction)))
                    stats.contacts+=current.contacts
                    stats.max_correction=max(stats.max_correction,current.max_correction)
                    contact_fraction=min(contact_fraction,fraction.value)
                    check(self.read_state(self.body,ptr(result),None,result.size))
                    actual_contact_correction=max(actual_contact_correction,float(np.linalg.norm(corrected-surface,axis=1).max()))
                    if self.contact_transport=='embedded_coupled':
                        check(self.lib.vhface_jaw_coupling_read(self.coupling,ptr(self.q)))
                    else:
                        moving=(self.bone[self.embed_ids]==1).any(1)
                        self.reaction+=self.generalized(surface[moving],reaction[moving]/dt)
                    if adaptive and contact_pass>=2:
                        audit=corrected.copy();remaining=ContactStats()
                        check(self.lib.lightphysics_surface_contact_project(self.contact,ptr(before),ptr(audit),dt,
                              .0002,1,None,ct.byref(remaining)))
                        residual=remaining.max_correction
                        local_crossings=self.intersections(corrected)
                        export_crossings=self.intersections(corrected.astype(np.float32).astype(np.float64))
                        if residual<=.00005 and not local_crossings and not export_crossings:break
                    continue
                check(self.lib.lightphysics_surface_contact_project(self.contact, ptr(before), ptr(corrected), dt,
                      .0002, 1, ptr(reaction), ct.byref(current)))
                stats.contacts += current.contacts
                stats.max_correction = max(stats.max_correction, current.max_correction)
                delta = (corrected - surface) / np.maximum(self.surface_mass[:, None], 1e-30)
                nodal=np.zeros_like(result)
                for k in range(4):
                    ids = self.embed_ids[:, k]
                    np.add.at(nodal, ids, delta * (self.embed_weights[:, k]*self.mass[ids])[:, None])
                changed=np.flatnonzero(np.any(nodal!=0,axis=1)).astype(np.int32)
                fraction=D(1.)
                if len(changed):
                    correction=np.ascontiguousarray(nodal[changed])
                    check(self.lib.lightphysics_deformable_project_guarded_batch(self.body,ptr(changed),ptr(correction),len(changed),ct.byref(fraction)))
                    if self.state_precision=='float64':check(self.read_state(self.body,ptr(result),None,result.size))
                    else:result+=fraction.value*nodal
                contact_fraction=min(contact_fraction,fraction.value)
                actual_contact_correction=max(actual_contact_correction,float(np.linalg.norm(self.surface(result)-surface,axis=1).max()))
                # Reaction on rigid dental vertices is returned even though
                # they have zero inverse mass in the tissue solve.
                moving = (self.bone[self.embed_ids] == 1).any(1)
                self.reaction += fraction.value*self.generalized(surface[moving], reaction[moving] / dt)
            # Check the transported, exported surface rather than assuming
            # that projection of an intermediate collision proxy converged.
            surface=np.ascontiguousarray(self.surface(result))
            corrected=surface.copy();remaining=ContactStats()
            check(self.lib.lightphysics_surface_contact_project(self.contact,ptr(before),ptr(corrected),dt,
                  .0002,1,None,ct.byref(remaining)))
            residual=remaining.max_correction
            if residual>.00005 or local_crossings or export_crossings:
                diagnostic=self.failure_geometry(result,dt,surface_correction=corrected-surface,contact_fraction=contact_fraction)
                diagnostic['contact_residual_mm']=float(residual*1000)
                diagnostic.update(proper_surface_crossings=local_crossings,float32_surface_crossings=export_crossings)
                raise GeometryStepError('full-face surface contact failed' if local_crossings or export_crossings else
                    'full-face contact residual exceeds 0.05 mm',diagnostic)
        ratio = volumes(result, self.tets) / self.rest_volume
        if not np.isfinite(result).all() or not np.isfinite(ratio).all() or ratio.min() < .0499:
            raise GeometryStepError('full-face step violated positive volume',self.failure_geometry(result,dt))
        x = np.ascontiguousarray(result, self.state_dtype)
        velocity = np.ascontiguousarray((result - previous) / dt * .98**(240*dt), self.state_dtype)
        if self.contact_transport=='embedded_coupled':self.speed[:]=(self.q-old_q)/dt
        check(self.write_state(self.body, ptr(x), ptr(velocity), x.size))
        self.positions[:] = result
        return dict(min_volume_ratio=float(ratio.min()), contacts=int(stats.contacts),
                    contact_volume_fraction_min=contact_fraction,
                    max_contact_correction_mm=actual_contact_correction * 1000,
                    max_contact_proposal_mm=stats.max_correction * 1000,
                    contact_residual_mm=residual*1000)

    def step(self, excitation, dt, *, subdivisions=3):
        if not math.isfinite(dt) or not 0 < dt <= .01:
            raise ValueError('full-face dt must be in (0,.01]')
        if self.cancel is not None and self.cancel.is_set():
            raise RuntimeError('biomechanical simulation cancelled')
        with Snapshot(self) as snapshot:
            try:
                return self._step(excitation, dt)
            except FloatingPointError:
                snapshot.restore()
                if subdivisions <= 0:
                    raise
                self.subdivision_retries+=1
                from .kinematic_targets import KinematicTargets
                kinematics=KinematicTargets(self)
                try:
                    kinematics.apply(.5)
                    first = self.step(excitation, dt / 2, subdivisions=subdivisions - 1)
                    kinematics.apply(1.)
                    second = self.step(excitation, dt / 2, subdivisions=subdivisions - 1)
                except BaseException:
                    snapshot.restore()
                    raise
                return dict(min_volume_ratio=min(first['min_volume_ratio'], second['min_volume_ratio']),
                            contact_volume_fraction_min=min(first['contact_volume_fraction_min'],second['contact_volume_fraction_min']),
                            contacts=first['contacts'] + second['contacts'],
                            max_contact_correction_mm=max(first['max_contact_correction_mm'], second['max_contact_correction_mm']),
                            max_contact_proposal_mm=max(first['max_contact_proposal_mm'],second['max_contact_proposal_mm']),
                            contact_residual_mm=max(first['contact_residual_mm'],second['contact_residual_mm']))
            except BaseException:
                snapshot.restore()
                raise

    def advance(self, excitation, seconds):
        if not math.isfinite(seconds) or seconds<=0:
            raise ValueError('full-face interval must be finite and positive')
        steps=seconds*240
        nearest=round(steps)
        # Timestamp subtraction at 30 Hz can yield 8.000000000000002
        # nominal substeps. It must not silently add a ninth physics step.
        count=max(1,nearest if abs(steps-nearest)<=1e-10*max(1,abs(steps)) else math.ceil(steps))
        start_steps,start_retries=self.native_steps,self.subdivision_retries
        with Snapshot(self) as initial:
            try:
                from .kinematic_targets import KinematicTargets
                kinematics=KinematicTargets(self)
                reports=[]
                for index in range(count):
                    kinematics.apply((index+1)/count)
                    reports.append(self.step(excitation, seconds/count))
            except BaseException:
                initial.restore()
                raise
        return dict(min_volume_ratio=min(r['min_volume_ratio'] for r in reports),
                    native_steps=self.native_steps-start_steps,nominal_substeps=count,
                    subdivision_retries=self.subdivision_retries-start_retries,
                    contact_volume_fraction_min=min(r['contact_volume_fraction_min'] for r in reports),
                    contacts=sum(r['contacts'] for r in reports),
                    max_contact_correction_mm=max(r['max_contact_correction_mm'] for r in reports),
                    max_contact_proposal_mm=max(r['max_contact_proposal_mm'] for r in reports),
                    contact_residual_mm=max(r['contact_residual_mm'] for r in reports))
