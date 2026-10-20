"""Offline full-face inverse muscle control and exact CPU geometry bakes."""
from __future__ import annotations

import argparse
from contextlib import closing,nullcontext
import hashlib
import json
import math
import os
import resource
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import numpy as np

from . import anatomy, articulation, rigdef
from .biomechanics_native import ROOT, LIBRARY, Session, Snapshot, volumes, NATIVE_VOLUME_FLOOR,freeze_library


def python_source_receipt():
    """Record the Python implementation present when a bake starts."""
    names=('anatomy','articulation','biomechanics','biomechanics_contacts',
           'biomechanics_mapping','biomechanics_native','biomechanics_trials',
           'jaw_model','kinematic_targets','rigdef','tongue_cables')
    paths=[Path(__file__).resolve().with_name(name+'.py') for name in names]
    return {str(path.relative_to(ROOT)):anatomy.digest(path) for path in paths}


def control_prior(meta, controls):
    result=np.zeros(len(meta['muscles']))
    for i,(name,links) in enumerate(zip(meta['muscles'],meta['control_prior'])):
        result[i]=sum(float(controls.get(c,0))*w for c,w in links.items())
        if name.startswith(('masseter_','temporalis_','medial_pterygoid_')):
            result[i]+=.03
    return np.clip(result,0,1)


def inverse_samples(arrays, meta, target, *, part_samples=80, feature_samples=64, moving_samples=48,eye_samples=64):
    """Cover every part, lip/eyelid features, and moving frontal skin."""
    rest=np.asarray(arrays['surface_rest'])
    selected=[]
    for part in meta['parts']:
        ids=np.arange(part['surface_start'],part['surface_start']+part['surface_count'])
        selected.extend(ids[::max(1,len(ids)//part_samples)])
    if 'fit_importance' not in arrays and 'fit_eye_features' not in arrays:
        return np.unique(selected).astype(np.int32)
    importance=arrays.get('fit_importance',np.ones(len(rest)))
    def spread(candidates,limit,weights=None):
        candidates=np.asarray(candidates,np.int32)
        if not len(candidates) or limit<=0:return []
        if len(candidates)<=limit:return candidates.tolist()
        p=rest[candidates];score=np.ones(len(p)) if weights is None else np.asarray(weights)
        distance=np.full(len(p),np.inf);result=[]
        next_id=int(np.argmax(score*np.square(p-p.mean(0)).sum(1)))
        for _ in range(limit):
            result.append(int(candidates[next_id]))
            distance=np.minimum(distance,np.square(p-p[next_id]).sum(1))
            distance[next_id]=-1
            next_id=int(np.argmax(distance*score))
        return result
    selected.extend(spread(np.flatnonzero(importance>1),feature_samples))
    if 'fit_eye_features' in arrays:
        selected.extend(spread(np.flatnonzero(arrays['fit_eye_features']),eye_samples))
    if 'fit_importance' not in arrays:return np.unique(selected).astype(np.int32)
    displacement=np.linalg.norm(np.asarray(target)-rest,axis=1)
    exterior=np.zeros(len(rest),bool)
    for part in meta['parts']:
        if part['name']=='skin':
            begin=part['surface_start'];exterior[begin:begin+part['surface_count']]=True
    exterior &= importance>0
    landmarks=meta.get('landmarks',{})
    if 'mouth' in landmarks and 'eye_y' in landmarks:
        mouth=np.asarray(landmarks['mouth']);height=max(.001,abs(float(landmarks['eye_y'])-mouth[1]))
        exterior &= (rest[:,1]>=mouth[1]-.6*height)&(rest[:,2]>=mouth[2]-.7*height)
    candidates=np.flatnonzero(exterior&(displacement>.00005))
    candidates=candidates[np.argsort(-displacement[candidates],kind='stable')[:512]]
    selected.extend(spread(candidates,moving_samples,displacement[candidates]**2))
    return np.unique(selected).astype(np.int32)


def regional_target_metrics(arrays,meta,target,surface):
    """Keep small oral/expressive regions from disappearing in a head average."""
    rest=arrays['surface_rest'];regions={}
    for part in meta['parts']:
        begin=part['surface_start'];ids=np.arange(begin,begin+part['surface_count'])
        regions[part['name']]=ids
        if part['name']=='skin':
            moving=np.linalg.norm(target[ids]-rest[ids],axis=1)>=.0005
            if 'fit_importance' in arrays:moving &= arrays['fit_importance'][ids]>0
            if moving.any():regions['moving_skin']=ids[moving]
    if 'fit_importance' in arrays:
        ids=np.flatnonzero(arrays['fit_importance']>1)
        if len(ids):regions['lip_features']=ids
    if 'fit_eye_features' in arrays:
        ids=np.flatnonzero(arrays['fit_eye_features'])
        if len(ids):regions['eyelid_features']=ids
    result={}
    for name,ids in regions.items():
        if not len(ids):continue
        neutral=float(np.sqrt(np.mean((rest[ids]-target[ids])**2))*1000)
        error=float(np.sqrt(np.mean((surface[ids]-target[ids])**2))*1000)
        # Resolve a meaningful fraction of an observable motion request, in
        # addition to the absolute 2 mm coordinate RMS bound. Submillimetre
        # targets retain a 0.25 mm error floor rather than a relative zero test.
        limit=min(2.,max(.25,.75*neutral))
        result[name]=dict(rms_mm=error,neutral_rms_mm=neutral,limit_mm=limit,passed=error<=limit)
    return result


def surface_area_metrics(rest,surface,faces):
    """Rotation-independent collapse check on complete exported triangles."""
    def areas(points):
        p=points[faces]
        return np.linalg.norm(np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]),axis=1)
    original=areas(rest);current=areas(surface)
    valid=original>1e-24
    ratios=current[valid]/original[valid]
    return dict(min_area_ratio=float(ratios.min()) if len(ratios) else 0.,
                collapsed_faces=int(np.count_nonzero(ratios<.05)),
                degenerate_source_faces=int(np.count_nonzero(~valid)),
                passed=bool(len(ratios) and valid.all() and np.isfinite(ratios).all() and (ratios>=.05).all()))


def inverse_step(state, target, prior, seconds, *, iterations=2, sample_ids=None, previous=None, cancel=None, progress=None,
                 trial_pool=None, sensitivities=None, extra_residual=None, extra_weight=1.,probe_substeps=0,stop=None,validate_trial=None,warm_start=None):
    """Bounded regularized native finite-difference inverse dynamics.

    All candidates start from the identical native/muscle/jaw snapshot. The
    accepted state is committed from a complete validated trial snapshot.
    """
    from scipy.optimize import lsq_linear
    target=np.asarray(target,np.float64)
    if target.shape!=state.surface_rest.shape or not np.isfinite(target).all():
        raise ValueError('invalid inverse-control surface target')
    if not math.isfinite(seconds) or not 0<seconds<=.2 or type(iterations) is not int or not 0<=iterations<=20:
        raise ValueError('invalid inverse-control interval/iterations')
    if sample_ids is None:
        sample_ids=np.arange(0,len(target),max(1,len(target)//256))
    ids=np.asarray(sample_ids)
    if ids.ndim!=1 or not len(ids) or ids.dtype.kind not in 'iu' or (ids<0).any() or (ids>=len(target)).any():
        raise ValueError('invalid inverse-control surface samples')
    ids=ids.astype(np.int32)
    prior=np.asarray(prior,np.float64)
    previous=np.asarray(previous if previous is not None else prior,np.float64)
    warm_start=np.asarray(warm_start,np.float64) if warm_start is not None else None
    history=(prior,previous) if warm_start is None else (prior,previous,warm_start)
    if any(v.shape!=(state.nm,) or not np.isfinite(v).all() or (v<0).any() or (v>1).any() for v in history):
        raise ValueError('invalid inverse-control muscle history/prior')
    if trial_pool is not None and trial_pool.owner is not state:
        raise ValueError('inverse trial pool belongs to a different session')
    if sensitivities is not None:
        from .biomechanics_trials import SensitivityCache
        if not isinstance(sensitivities,SensitivityCache):raise ValueError('invalid muscle sensitivity cache')
    from .biomechanics_trials import validate_probe_substeps,response_gains
    validate_probe_substeps(probe_substeps)
    if stop is not None and not callable(stop):raise ValueError('invalid inverse stopping callback')
    if validate_trial is not None and not callable(validate_trial):raise ValueError('invalid inverse geometry callback')
    if not math.isfinite(extra_weight) or extra_weight<0:raise ValueError('invalid extra objective weight')
    point_count=3*len(ids)
    initial_extra=np.asarray(extra_residual(state.surface()),np.float64) if extra_residual is not None and extra_weight else np.empty(0)
    if initial_extra.ndim!=1 or not np.isfinite(initial_extra).all():raise ValueError('invalid extra geometry residual')
    extra_count=len(initial_extra)
    extra_scale=math.sqrt(extra_weight*point_count/extra_count) if extra_count else 0.
    def extra_values(surface):
        values=np.asarray(extra_residual(surface),np.float64)
        if values.shape!=(extra_count,) or not np.isfinite(values).all():raise ValueError('invalid extra geometry residual')
        return values
    def residual_for(surface):
        points=(surface[ids]-target[ids]).reshape(-1)/.001
        return np.concatenate((points,extra_values(surface)*extra_scale)) if extra_count else points
    # A zero authored prior must not prevent the inverse solve discovering an
    # antagonist or a combination missing from a hand-authored control map.
    active=np.arange(state.nm)
    chosen=(warm_start if warm_start is not None else prior).copy()
    evaluations=0;unsafe_trials=0;seed_fallback=False;unsafe_trial_diagnostics=[];geometry_rejections=0
    sensitivity_reused=False;sensitivity_refreshes=0;sensitivity_fallbacks=0
    probe_calibrations=[]
    early_stopped=False;iterations_completed=0
    initial_activation=state.activation.copy()
    def native_sessions():return [state]+(trial_pool.sessions if trial_pool is not None else [])
    initial_native_steps=sum(s.native_steps for s in native_sessions())
    initial_retries=sum(s.subdivision_retries for s in native_sessions())
    with Snapshot(state) as initial:
        best_state=None;best_stats=None;last_stats=None;last_surface=None
        def evaluate(excitation):
            nonlocal evaluations,unsafe_trials,last_stats,last_surface,geometry_rejections
            if cancel is not None and cancel.is_set():
                raise RuntimeError('biomechanical inverse fitting cancelled')
            initial.restore()
            try:
                last_stats=state.advance(excitation,seconds)
                last_surface=state.surface()
                if validate_trial is not None:
                    try:validate_trial(last_surface)
                    except FloatingPointError:geometry_rejections+=1;raise
            except FloatingPointError as exc:
                unsafe_trials+=1
                if len(unsafe_trial_diagnostics)<2:
                    evidence=dict(error=str(exc))
                    if hasattr(exc,'diagnostics'):evidence['failed_step']=exc.diagnostics
                    unsafe_trial_diagnostics.append(evidence)
                raise
            evaluations+=1
            return residual_for(last_surface)
        def cost(residual,x):
            return float(np.sum(residual**2)/point_count+.01*np.mean((x-prior)**2)+.03*np.mean((x-previous)**2))
        def retain_trial():
            nonlocal best_state,best_stats
            candidate=Snapshot(state)
            if best_state is not None:
                best_state.close()
            best_state=candidate
            best_stats=dict(last_stats)
        try:
            seeds=[]
            for name,value in ([('warm_start',warm_start)] if warm_start is not None else [])+[
                    ('authored_prior',prior),('previous_drive',previous)]:
                if not any(np.array_equal(value,v) for _,v in seeds):seeds.append((name,value))
            for seed_index,(seed_source,value) in enumerate(seeds):
                chosen=value.copy()
                try:residual=evaluate(chosen)
                except FloatingPointError:
                    if seed_index+1==len(seeds):raise
                    seed_fallback=True
                else:break
            retain_trial()
            before=float(np.sqrt(np.mean(residual[:point_count]**2)))
            extra_before=float(np.sqrt(np.mean((residual[point_count:]/extra_scale)**2))) if extra_count else None
            objective=cost(residual,chosen)
            objective_before=objective
            for iteration in range(iterations):
                if stop is not None and stop(state.surface(best_state.positions)):
                    early_stopped=True
                    if progress:progress(1.,'validated target tolerances reached')
                    break
                if not len(active):
                    break
                iterations_completed+=1
                def sensitivity_progress(column,total):
                    if progress:
                        progress((iteration+.8*column/total)/iterations,
                                 f'muscle sensitivity {column}/{total}, iteration {iteration+1}/{iterations}')
                def fresh_jacobian():
                    nonlocal evaluations,unsafe_trials,sensitivity_refreshes,chosen,residual,objective
                    columns=[];full_columns=[]
                    base_surface=state.surface(best_state.positions)
                    deltas=np.where(chosen[active]<=.98,.02,-.02)
                    probes=np.tile(chosen,(len(active),1))
                    probes[np.arange(len(active)),active]+=deltas
                    steps=seconds*240;nearest=round(steps)
                    full_steps=max(1,nearest if abs(steps-nearest)<=1e-10*max(1,abs(steps)) else math.ceil(steps))
                    shortened=bool(probe_substeps and probe_substeps<full_steps)
                    probe_seconds=seconds*probe_substeps/full_steps if shortened else seconds
                    if shortened:
                        initial.restore()
                        try:
                            state.advance(chosen,probe_seconds)
                        except FloatingPointError:
                            # A full-frame safe seed does not imply that a
                            # different partial trajectory is admissible.
                            shortened=False;probe_seconds=seconds
                        else:
                            evaluations+=1;short_base=state.surface()
                    def collect(surface,delta):
                        if surface is None:
                            if not shortened:columns.append(np.zeros_like(residual))
                            full_columns.append(np.zeros_like(base_surface))
                        elif shortened:
                            full_columns.append((surface-short_base)/.001/delta)
                        else:
                            columns.append((residual_for(surface)-residual)/delta)
                            full_columns.append((surface-base_surface)/.001/delta)
                    if trial_pool is None:
                        for column,(probe,delta) in enumerate(zip(probes,deltas)):
                            sensitivity_progress(column+1,len(active))
                            if cancel is not None and cancel.is_set():
                                raise RuntimeError('biomechanical inverse fitting cancelled')
                            try:
                                if shortened:
                                    initial.restore();state.advance(probe,probe_seconds);evaluations+=1
                                    collect(state.surface(),delta)
                                else:
                                    evaluate(probe);collect(last_surface,delta)
                            except FloatingPointError:
                                if shortened:unsafe_trials+=1
                                collect(None,delta)
                    else:
                        surfaces=trial_pool.surfaces(initial,probes,probe_seconds,cancel=cancel,progress=sensitivity_progress)
                        for surface,delta in zip(surfaces,deltas):
                            if surface is None:
                                unsafe_trials+=1;collect(None,delta)
                            else:
                                evaluations+=1
                                collect(surface,delta)
                    if shortened:
                        gains=response_gains(state.muscles,initial_activation,chosen,deltas,seconds/full_steps,full_steps,probe_substeps)
                        full_columns=np.asarray(full_columns)*gains[:,None,None]
                        calibration_scale=1.;calibration_cosine=None;calibration_valid=False
                        native_column=None
                        approximate=np.column_stack([(residual_for(base_surface+.001*delta*derivative)-residual)/delta
                                                     for derivative,delta in zip(full_columns,deltas)])
                        gradient=approximate.T@residual
                        direction=np.clip(chosen+np.where(gradient<0,.02,-.02),0,1)-chosen
                        descent=-gradient*direction;index=int(np.argmax(descent));accepted_calibration=False
                        # Calibrate one target-aligned response. Exciting every
                        # unrelated tongue/jaw family together is expensive and
                        # can invalidate a local linear response unnecessarily.
                        if descent[index]>1e-12:
                            trial=chosen.copy();trial[index]+=direction[index]
                            predicted=(direction[index]*full_columns[index])[ids].reshape(-1)
                            try:
                                candidate=evaluate(trial)
                            except FloatingPointError:
                                pass
                            else:
                                native_column=(last_surface-base_surface)/.001/direction[index]
                                observed=((last_surface-base_surface)/.001)[ids].reshape(-1)
                                energy=float(predicted@predicted)
                                norm=float(np.linalg.norm(predicted)*np.linalg.norm(observed))
                                if energy>1e-20 and norm>1e-20:
                                    calibration_cosine=float(predicted@observed/norm)
                                    if calibration_cosine>0:
                                        calibration_scale=float(np.clip((predicted@observed)/energy,.001,1000.))
                                        calibration_valid=True
                                score=cost(candidate,trial)
                                if score<objective-1e-9:
                                    # Keep this full-duration, validated native
                                    # improvement even if a later Newton step fails.
                                    chosen,residual,objective=trial,candidate,score
                                    retain_trial();accepted_calibration=True
                        full_columns*=calibration_scale
                        if native_column is not None:full_columns[index]=native_column
                        for derivative,delta in zip(full_columns,deltas):
                            virtual=base_surface+.001*delta*derivative
                            columns.append((residual_for(virtual)-residual_for(base_surface))/delta)
                        probe_calibrations.append(dict(substeps=probe_substeps,seconds=probe_seconds,
                            full_substeps=full_steps,directional_scale=calibration_scale,
                            directional_cosine=calibration_cosine,directional_calibration_valid=calibration_valid,
                            calibration_muscle=int(index) if descent[index]>1e-12 else None,
                            calibration_accepted=accepted_calibration,calibration_policy='target_aligned_single_muscle',
                            full_native_column_refined=native_column is not None,
                            model='short_native_response+activation_inertial_gain+full_native_directional_calibration'))
                    if sensitivities is not None:
                        sensitivities.refresh(state,seconds,full_columns)
                    sensitivity_refreshes+=1
                    return np.column_stack(columns)
                def try_update(jacobian):
                    nonlocal chosen,residual,objective
                    jacobian=jacobian/math.sqrt(point_count)
                    regularization=math.sqrt(.04/max(1,len(active)))
                    matrix=np.vstack((jacobian,regularization*np.eye(len(active))))
                    preferred=(.01*prior[active]+.03*previous[active])/.04
                    rhs=np.concatenate((-residual/math.sqrt(point_count),regularization*(preferred-chosen[active])))
                    delta=lsq_linear(matrix,rhs,bounds=(-chosen[active],1-chosen[active]),method='bvls',tol=1e-6).x
                    for factor in (1.,.5,.25):
                        trial=chosen.copy();trial[active]=np.clip(trial[active]+factor*delta,0,1)
                        if np.array_equal(trial,chosen):continue
                        try:
                            candidate=evaluate(trial)
                        except FloatingPointError:
                            continue
                        score=cost(candidate,trial)
                        if score<objective-1e-9:
                            chosen,residual,objective=trial,candidate,score
                            retain_trial()
                            return True
                    return False
                if sensitivities is not None and sensitivities.compatible(state,seconds):
                    sensitivity_reused=True
                    if progress:progress(iteration/iterations,'reusing muscle sensitivities; validating native trial')
                    jacobian=sensitivities.jacobian(ids)
                    if extra_count:
                        base_surface=state.surface(best_state.positions);baseline=extra_values(base_surface)
                        columns=[]
                        for index in active:
                            delta=.02 if chosen[index]<=.98 else -.02
                            virtual=base_surface+delta*.001*sensitivities.derivatives[index]
                            columns.append((extra_values(virtual)-baseline)*extra_scale/delta)
                        jacobian=np.vstack((jacobian,np.column_stack(columns)))
                    accepted=try_update(jacobian)
                    if not accepted:
                        sensitivity_fallbacks+=1
                        sensitivities.clear()
                        accepted=try_update(fresh_jacobian())
                else:
                    accepted=try_update(fresh_jacobian())
                if not accepted:
                    break
            best_state.restore()
            stats=best_stats
            if progress:
                progress(1.,'muscle solve accepted')
        except BaseException:
            initial.restore()
            if sensitivities is not None:sensitivities.clear()
            raise
        finally:
            if best_state is not None:
                best_state.close()
    if sensitivities is not None and sensitivities.owner is state and not sensitivity_refreshes:sensitivities.uses+=1
    return chosen,dict(stats,target_rms_before_mm=before,
                       target_rms_after_mm=float(np.sqrt(np.mean(residual[:point_count]**2))),evaluations=evaluations,
                       objective_before=objective_before,objective_after=objective,
                       extra_objective_terms=extra_count,extra_objective_weight=extra_weight if extra_count else 0.,
                       extra_objective_rms_before=extra_before,
                       extra_objective_rms_after=float(np.sqrt(np.mean((residual[point_count:]/extra_scale)**2))) if extra_count else None,
                       candidate_muscles=len(active),active_muscles=int(np.count_nonzero(state.activation>1e-6)),
                       excited_muscles=int(np.count_nonzero(chosen>1e-6)),unsafe_trials=unsafe_trials,seed_fallback=seed_fallback,
                       unsafe_trial_diagnostics=unsafe_trial_diagnostics,
                       geometry_rejections=geometry_rejections,
                       seed_source=seed_source,warm_start_provided=warm_start is not None,
                       target_sample_count=len(ids),
                       inverse_workers=trial_pool.workers if trial_pool is not None else 1,
                       inverse_active_workers=len(trial_pool.sessions) if trial_pool is not None else 0,
                       sensitivity_reused=sensitivity_reused,sensitivity_refreshes=sensitivity_refreshes,
                       sensitivity_fallbacks=sensitivity_fallbacks,
                       sensitivity_probe_substeps=probe_substeps,sensitivity_probe_calibrations=probe_calibrations,
                       native_steps_evaluated=sum(s.native_steps for s in native_sessions())-initial_native_steps,
                       subdivision_retries_evaluated=sum(s.subdivision_retries for s in native_sessions())-initial_retries,
                       early_stopped=early_stopped,iterations_completed=iterations_completed,
                       accepted_state_source='saved_validated_native_trial',
                       target_is_measured_ground_truth=False)


def target_sequence(rig_dir,arrays,meta,frames,aux=None,*,motion=None,speech_options=None):
    from .gnm_model import GNMModel
    from .gnm_expression import evaluate
    from ..reconstruction.gnm_motion import rig_bind
    rig_dir=Path(rig_dir)
    model=GNMModel()
    bind=rig_bind(rig_dir)
    fitted_bind=bind['bind_residual']+arrays.get('bind_delta',0)
    definition=json.loads((rig_dir/'rig.json').read_text())
    mapping=definition.get('face_model_source',{}).get('expression_mapping')
    if mapping is None:
        raise ValueError('full biomechanics needs a geometry-matched GNM expression map')
    if 'expression_coefficients' in arrays:
        mapping=dict(mapping,coefficients=arrays['expression_coefficients'],controls=meta['expression_controls'])
    # A verified fitted performance takes priority over approximate semantic
    # controls. Both are targets; neither is added to the simulated result.
    ids=arrays['source_ids']
    times=np.array([f['t'] for f in frames])
    options=speech_options or {}
    profile=articulation.build(aux,times,strength=float(options.get('speech_strength',1))) if aux is not None else None
    if profile is not None:
        from .speech import validate_emotions,_emotion_at
        emotion_keys=validate_emotions(options.get('emotion_keyframes'),float(aux['duration']))
    use_native=motion is None
    from .biomechanics_mapping import load_native_rig,capture_transforms,load_skinning,skin_vectors
    exterior=np.flatnonzero(model.group('skin_exterior'))
    exterior_out=np.flatnonzero(model.group('skin_exterior')[ids])
    exterior_map=np.searchsorted(exterior,ids[exterior_out])
    local_rig=rigdef.Rig(definition)
    control_names=[c['name'] for c in definition['controls']]
    skinning=(load_skinning(rig_dir,len(exterior)) if use_native and
              ('expression_coefficients' in arrays or 'bind_delta' in arrays) else None)
    with closing(load_native_rig(rig_dir)) if use_native else nullcontext(None) as teacher:
        for index,frame in enumerate(frames):
            controls=dict(frame['v'])
            if profile is not None:
                controls=articulation.compose_controls(controls,profile['values'][index],
                    emotions=_emotion_at(emotion_keys,float(frame['t'])),emotion_strength=float(options.get('emotion_strength',.6)))
            expression=(motion['expression'][index] if motion is not None else evaluate(mapping,controls))
            rotations=motion['rotations'][index].copy() if motion is not None else None
            translation=(motion['translation'][index] if 'translation' in motion else np.zeros(3)) if motion is not None else None
            points,_=model.evaluate(bind['identity'],expression,rotations,translation,bind_residual=fitted_bind)
            points=float(bind['scale'])*points@bind['alignment'].T+bind['offset']
            if motion is not None:
                transforms=(motion['transforms'][index] if 'transforms' in motion else
                            capture_transforms(model,bind,rotations,translation))
                head=transforms['head']
                # Factor the complete neck/head affine transform out of the
                # target. Eye rotations and relative neck deformation remain.
                inverse_head=np.linalg.inv(head)
                points=points@inverse_head[:3,:3].T+inverse_head[:3,3]
            target=points[ids]
            if teacher is not None:
                # The morph-only PCA map cannot represent all joint motion.
                # Always retain evaluated joint geometry. Accepted expression
                # fits modify only its PCA component, never replace skinning.
                local=dict(controls,headYaw=0.,headPitch=0.,headRoll=0.)
                values=np.array([local.get(name,0.) for name in control_names],np.float32)
                posed=teacher.eval(values)
                target[exterior_out]=posed[exterior_map]
                transforms=local_rig.evaluate(local)['skin']
                if 'expression_coefficients' in arrays or 'bind_delta' in arrays:
                    original=definition['face_model_source']['expression_mapping']
                    # The calibrated seed already describes the native teacher.
                    # Only a subsequent capture fit is an additional correction.
                    # Packages predating the stored seed retain their old basis.
                    if 'expression_seed_coefficients' in arrays:
                        original=dict(original,coefficients=arrays['expression_seed_coefficients'],
                                      controls=meta['expression_controls'])
                    base,_=model.evaluate(bind['identity'],evaluate(original,controls),bind_residual=bind['bind_residual'])
                    base=float(bind['scale'])*base@bind['alignment'].T+bind['offset']
                    correction=skin_vectors(points[exterior]-base[exterior],skinning,transforms)
                    target[exterior_out]+=correction[exterior_map]
                jaw=transforms[local_rig.jidx['teeth_lower']]
                rest=arrays['surface_rest']
                lower=model.group('lower_teeth_and_gums')[ids]
                target[lower]=rest[lower]@jaw[:3,:3].T+jaw[:3,3]
                sock=model.group('mouth_sock')[ids]
                mouth_y=float(np.asarray(meta['landmarks']['mouth'])[1])
                influence=np.clip((mouth_y+.004-rest[sock,1])/.012,0,1)
                moved=rest[sock]@jaw[:3,:3].T+jaw[:3,3]
                target[sock]=(1-influence[:,None])*target[sock]+influence[:,None]*moved
                tongue=model.group('tongue')[ids]
                target[tongue]=rest[tongue]@jaw[:3,:3].T+jaw[:3,3]
                # GNM's morph-only expression map does not rotate its globes.
                # Match the exact semantic eye transforms used by the physical
                # rigid targets, rather than asking muscles to undo gaze.
                for part in meta.get('parts',[]):
                    if part['name'] not in ('eye_left','eye_right'):continue
                    side='L' if part['name']=='eye_left' else 'R'
                    eye=transforms[local_rig.jidx['eye_'+side]]
                    sample=np.arange(part['surface_start'],part['surface_start']+part['surface_count'])
                    target[sample]=rest[sample]@eye[:3,:3].T+eye[:3,3]
                if profile is None and tongue.any():
                    p=rest[tongue];length=max(.001,float(np.ptp(p[:,2])))
                    tip=np.clip((p[:,2]-p[:,2].min())/length,0,1)
                    target[tongue,0]+=.01*tip*(controls.get('tongueLeft',0)-controls.get('tongueRight',0))
                    target[tongue,1]+=.015*tip*controls.get('tongueUp',0)-.010*tip*controls.get('tongueDown',0)+.010*tip**2*controls.get('tongueCurlUp',0)
                    target[tongue,2]+=.024*tip*controls.get('tongueOut',0)*(.2+.8*controls.get('jawOpen',0))
            if profile is not None:
                target=articulation.geometric_targets(target,arrays,model,profile,index)
            yield controls,target,profile


def _mesh_samples(stream,part,arrays,cache,times,fps):
    start,count=part['surface_start'],part['surface_count']
    faces=arrays['faces'][part['face_start']:part['face_start']+part['face_count']]-start
    name=part['name']
    stream.write(f'    def Mesh "{name}" {{\n        uniform token subdivisionScheme = "none"\n')
    stream.write('        int[] faceVertexCounts = ['+', '.join(['3']*len(faces))+']\n')
    stream.write('        int[] faceVertexIndices = ['+', '.join(map(str,faces.reshape(-1)))+']\n')
    stream.write('        point3f[] points.timeSamples = {\n')
    for index,stamp in enumerate(times):
        points=cache[index,start:start+count]
        row=', '.join('('+', '.join(format(float(v),'.9g') for v in p)+')' for p in points)
        stream.write(f'            {stamp*fps:.12g}: [{row}],\n')
    stream.write('        }\n    }\n')


def export_usd(path,arrays,meta,cache,times,fps):
    """Author evaluated points, without skeleton bindings or extra transforms."""
    path=Path(path);partial=path.with_name(path.name+'.partial')
    with partial.open('w') as stream:
        stream.write('#usda 1.0\n(\n    defaultPrim = "Character"\n    metersPerUnit = 1\n    upAxis = "Y"\n')
        stream.write(f'    timeCodesPerSecond = {fps:.12g}\n    framesPerSecond = {fps:.12g}\n    startTimeCode = 0\n    endTimeCode = {times[-1]*fps:.12g}\n)\n')
        stream.write('def Xform "Character" {\n')
        for part in meta['parts']:
            _mesh_samples(stream,part,arrays,cache,times,fps)
        stream.write('}\n')
    partial.replace(path)


def verify_usd(path,meta,cache,times,fps):
    """Independent CPU reader; verify all parts and all exported samples."""
    from pxr import Usd,UsdGeom,UsdSkel
    stage=Usd.Stage.Open(str(path))
    if not stage or stage.GetTimeCodesPerSecond()!=fps:
        raise ValueError('USD bake time base failed round-trip')
    maximum=0.
    for part in meta['parts']:
        mesh=UsdGeom.Mesh.Get(stage,'/Character/'+part['name'])
        if not mesh or mesh.GetPrim().HasAPI(UsdSkel.BindingAPI):
            raise ValueError('USD evaluated mesh is missing or has an extra skin binding')
        first,count=part['surface_start'],part['surface_count']
        for index,stamp in enumerate(times):
            replay=np.asarray(mesh.GetPointsAttr().Get(float(stamp*fps)))
            reference=cache[index,first:first+count]
            if replay.shape!=reference.shape or not np.isfinite(replay).all():
                raise ValueError('USD bake contains missing/nonfinite geometry')
            maximum=max(maximum,float(np.max(abs(replay-reference))))
    if maximum>1e-6:
        raise ValueError('USD geometry differs from point cache by more than 1 micrometre')
    return maximum*1000


def lip_surface_aperture(rest,faces,upper,lower,cache):
    """Symmetric medial lip gaps, independent of opposing vertex alignment."""
    from ..reconstruction.contact_runtime import closest_on_triangles
    def distances(ids,triangles,positions):
        tri=positions[triangles]
        area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)
        tri=tri[area>1e-16]
        if not len(tri):
            raise ValueError('degenerate opposing lip surface geometry')
        _,_,distance=closest_on_triangles(positions[ids],tri[None,:,0],tri[None,:,1],tri[None,:,2])
        return distance.min(1)
    samples=[]
    for source,target in ((upper,lower),(lower,upper)):
        ids=np.flatnonzero(source)
        triangles=faces[target[faces].all(1)]
        if not len(ids) or not len(triangles):
            raise ValueError('missing opposing lip surface geometry')
        x=rest[ids,0];center=float(np.median(x));width=float(np.ptp(x))
        medial=ids[np.abs(x-center)<=.3*width+1e-12]
        if not len(medial):
            raise ValueError('missing medial lip samples')
        distance=distances(medial,triangles,rest)
        chosen=medial[np.argsort(distance,kind='stable')[:min(32,len(medial))]]
        samples.append((chosen,triangles))
    apertures=[]
    for positions in cache:
        direction=[]
        for chosen,triangles in samples:
            direction.extend(distances(chosen,triangles,positions))
        apertures.append(float(np.median(direction))*1000)
    return np.asarray(apertures)


def labiodental_surface_gap(arrays,upper,lower,teeth,cache):
    """Medial lower-lip seam to the frontal upper dental surface, in mm."""
    from ..reconstruction.contact_runtime import closest_on_triangles
    from .biomechanics_contacts import labiodental_samples
    rest=arrays['surface_rest'];faces=arrays['faces']
    ids,triangles=labiodental_samples(rest,faces,lower,teeth)
    if not len(ids) or not len(triangles):raise ValueError('missing labiodental contact geometry')
    output=[]
    for points in cache:
        p=points[triangles]
        area=np.linalg.norm(np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]),axis=1)
        p=p[area>1e-16]
        if not len(p):raise ValueError('degenerate upper dental surface')
        _,_,distance=closest_on_triangles(points[ids],p[None,:,0],p[None,:,1],p[None,:,2])
        output.append(float(np.median(distance.min(1)))*1000)
    return np.asarray(output)


def speech_contact_metrics(arrays,profile,cache,head_transforms):
    if profile is None:
        return dict(events=[],passed=True,source='no speech alignment')
    if profile.get('speech_strength',1)<.7:
        return dict(events=[],passed=True,source='full closure goals not requested at reduced speech strength')
    from .gnm_model import GNMModel
    from scipy.spatial import cKDTree
    model=GNMModel();source=arrays['source_ids'];rest=arrays['surface_rest']
    upper_mask=model.group('upper_lip')[source];lower_mask=model.group('lower_lip')[source]
    upper=np.flatnonzero(upper_mask);lower=np.flatnonzero(lower_mask)
    if not len(upper) or not len(lower):
        return dict(events=[],passed=False,source='missing opposing lip geometry')
    distance,nearest=cKDTree(rest[lower]).query(rest[upper])
    selected=np.argsort(distance)[:min(32,len(upper))]
    u,l=upper[selected],lower[nearest[selected]]
    # Retain the old vertex-pair proxy for comparison, but use actual surface
    # separation for acceptance. Rest-selected medial samples exclude corners.
    vertex_aperture=np.array([np.median(np.linalg.norm(p[u]-p[l],axis=1))*1000 for p in cache])
    try:
        aperture=lip_surface_aperture(rest,arrays['faces'],upper_mask,lower_mask,cache)
    except ValueError as error:
        return dict(events=[],passed=False,source=str(error))
    times=profile['times'];events=[];labiodental=None;labiodental_error=None
    intervals=profile['intervals']
    for index,item in enumerate(intervals):
        phone=articulation.normalized(item.get('articulator_phone',item['phone']))
        next_item=intervals[index+1] if index+1<len(intervals) else {}
        next_phone=articulation.normalized(next_item.get('articulator_phone',next_item.get('phone','')))
        bilabial=(phone in articulation.BILABIAL or (phone in ('cl','N') and next_phone in articulation.BILABIAL)
                  or (phone=='N' and next_phone in articulation.BILABIAL_FRICATIVES))
        dental=phone in articulation.LABIODENTAL or (phone=='cl' and next_phone in articulation.LABIODENTAL)
        if not bilabial and not dental:
            continue
        step=float(np.median(np.diff(times))) if len(times)>1 else 1/30
        window=np.flatnonzero((times>=item['start']-step)&(times<=item['end']+step))
        if not len(window):
            window=np.array([int(np.argmin(abs(times-.5*(item['start']+item['end']))))])
        if bilabial:
            best=int(window[np.argmin(aperture[window])])
            events.append(dict(phone=phone,start=item['start'],end=item['end'],closure_time=float(times[best]),
                               contact_type='bilabial',aperture_mm=float(aperture[best]),limit_mm=.2,
                               vertex_pair_aperture_mm=float(vertex_aperture[best]),passed=bool(aperture[best]<=.2)))
        else:
            if labiodental is None and labiodental_error is None:
                try:
                    teeth=(model.group('teeth')&model.group('upper_teeth_and_gums'))[source]
                    labiodental=labiodental_surface_gap(arrays,upper_mask,lower_mask,teeth,cache)
                except ValueError as error:
                    labiodental_error=str(error)
            event=dict(phone=phone,start=item['start'],end=item['end'],contact_type='labiodental',limit_mm=.5)
            if labiodental_error is not None:
                event.update(passed=False,reason=labiodental_error)
            else:
                best=int(window[np.argmin(labiodental[window])])
                event.update(contact_time=float(times[best]),gap_mm=float(labiodental[best]),
                             passed=bool(labiodental[best]<=.5))
            events.append(event)
    return dict(events=events,passed=all(e['passed'] for e in events),
                checked_contact_types=['bilabial','labiodental'],
                unvalidated_contact_types=['bilabial_fricative','alveolar','dental_tongue','velar'],
                source='rest-selected medial point-to-triangle lip and dental gaps vs aligned events')


def review_movie(out,arrays,meta,cache,fps,audio=None,*,cancel=None,size=192):
    from . import preview
    from PIL import Image
    if shutil.which('ffmpeg') is None:
        raise FileNotFoundError('FFmpeg is required for CPU motion review')
    out=Path(out);silent=out/'physics_review_silent.mp4'
    command=['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{size*3}x{size}',
             '-r',str(fps),'-i','pipe:0','-an','-c:v','libx264','-threads','1','-pix_fmt','yuv420p',str(silent)]
    center=arrays['surface_rest'].mean(0);extent=np.ptp(arrays['surface_rest'],axis=0).max()*.55
    faces=arrays['faces']
    oral=[faces[p['face_start']:p['face_start']+p['face_count']] for p in meta['parts']
          if p['name'] in ('tongue','gums_upper','teeth_upper','gums_lower','teeth_lower')]
    cut_faces=np.concatenate(oral) if oral else faces
    cut=cut_faces[(arrays['surface_rest'][cut_faces,0].mean(1)>=center[0])]
    cut=np.concatenate((cut,cut[:,::-1]))  # inspect interior-facing surfaces
    mouth=np.asarray(meta.get('landmarks',{}).get('mouth',center))
    mouth_ids=np.unique(cut_faces) if oral else np.argsort(np.linalg.norm(arrays['surface_rest']-mouth,axis=1))[:32]
    with (out/'physics_review.log').open('w') as log:
        proc=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=log)
        try:
            for i,positions in enumerate(cache):
                if cancel is not None and cancel.is_set():
                    raise RuntimeError('CPU motion review cancelled')
                panels=[preview.render(positions,faces,size=size,yaw=yaw,center=center,extent=extent,wire=True) for yaw in (0,75)]
                panels.append(preview.render(positions,cut,size=size,yaw=75,center=positions[mouth_ids].mean(0),extent=extent*.42,wire=True))
                image=np.concatenate(panels,axis=1)
                if i==0:
                    Image.fromarray(image).save(out/'physics_preview.png')
                proc.stdin.write(image.tobytes())
            proc.stdin.close()
            if proc.wait():
                raise RuntimeError('CPU review encoding failed; inspect physics_review.log')
        except BaseException:
            proc.kill();proc.wait()
            raise
    final=out/'physics_review.mp4'
    if audio is not None and Path(audio).is_file():
        command=['ffmpeg','-v','error','-y','-i',str(silent),'-i',str(audio),'-map','0:v:0','-map','1:a:0',
                 '-c:v','copy','-c:a','aac','-threads','1','-shortest',str(final)]
        from ..process import run
        status=run(command,cancel=cancel or threading.Event(),on_line=lambda _:None,cwd=ROOT,
                   env=dict(os.environ,TMPDIR=str(out.resolve())))
        if status:
            raise RuntimeError('CPU review audio mux failed')
        silent.unlink()
    else:
        silent.replace(final)


def simulate(rig_dir,take_dir,*,out=None,package=None,threads=4,iterations=2,inverse_workers=1,
             sensitivity_reuse_frames=0,sensitivity_probe_substeps=0,state_precision='auto',mechanics_solver='partitioned',review=True,
             cancel=None,progress=None,library_path=LIBRARY):
    from ..reconstruction.gnm_fit import cpu_threads
    python_sources_at_start=python_source_receipt()
    rig_dir,take_dir=Path(rig_dir),Path(take_dir)
    if type(iterations) is not int or not 0<=iterations<=20:
        raise ValueError('inverse iterations must be 0..20')
    from .biomechanics_trials import NativeTrialPool,SensitivityCache,validate_workers,validate_reuse_frames,validate_probe_substeps,validate_state_precision,validate_mechanics_solver
    validate_workers(inverse_workers,threads)
    validate_reuse_frames(sensitivity_reuse_frames)
    validate_probe_substeps(sensitivity_probe_substeps)
    validate_state_precision(state_precision)
    validate_mechanics_solver(mechanics_solver,state_precision)
    mechanics_options=(dict(jaw_mode='coupled_volume' if mechanics_solver=='coupled' else 'coupled_block',
                            contact_transport='embedded_coupled') if mechanics_solver!='partitioned' else {})
    arrays,meta=anatomy.load(rig_dir,package)
    metadata_sha256=hashlib.sha256(json.dumps(meta,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    animation=json.loads((take_dir/'animation.json').read_text())
    if animation.get('format')!='vhuman.performance.v1':
        raise ValueError('full biomechanics requires a vhuman.performance.v1 take')
    frames=animation['frames'];fps=float(animation['fps'])
    if not 1<=len(frames)<=900 or not math.isfinite(fps) or not 10<=fps<=60:
        raise ValueError('full biomechanics accepts 1–900 frames at 10–60 FPS')
    times=np.array([frame['t'] for frame in frames],np.float64)
    if not np.isfinite(times).all() or times[0]!=0 or (np.diff(times)<=0).any() or (np.diff(times)> .2).any():
        raise ValueError('invalid full-face take timestamps')
    for frame in frames:
        if not isinstance(frame.get('v'),dict) or any(k not in rigdef.CONTROLS or not math.isfinite(v) for k,v in frame['v'].items()):
            raise ValueError('invalid full-face take controls')
    aux=json.loads((take_dir/'align.json').read_text()) if (take_dir/'align.json').is_file() else None
    motion=None
    if (take_dir/'motion.json').is_file() and aux is None:
        receipt=json.loads((take_dir/'motion.json').read_text())
        if receipt.get('motion_sha256')!=anatomy.digest(take_dir/'motion.npz'):
            raise ValueError('native capture motion hash mismatch')
        if not receipt.get('quality_gate'):
            raise ValueError('native capture did not pass its held-out fitting gate')
        with np.load(take_dir/'motion.npz',allow_pickle=False) as z:
            if len(z['expression'])!=len(frames) or not np.allclose(z['timestamps'],times,atol=1e-6):
                raise ValueError('native capture/control timestamps differ')
            from ..reconstruction.gnm_motion import rig_bind
            if not np.array_equal(z['identity'],rig_bind(rig_dir)['identity']):
                raise ValueError('native capture identity differs from anatomy')
            if z['rotations'].shape!=(len(frames),4,3) or not np.isfinite(z['rotations']).all():
                raise ValueError('invalid native capture joint rotations')
            translation=z['translation'].copy() if 'translation' in z else np.zeros((len(frames),3))
            if translation.shape!=(len(frames),3) or not np.isfinite(translation).all():
                raise ValueError('invalid native capture translations')
            motion=dict(expression=z['expression'].copy(),rotations=z['rotations'].copy(),translation=translation)
            from .gnm_model import GNMModel
            from .biomechanics_mapping import capture_transforms
            model=GNMModel();bind=rig_bind(rig_dir)
            motion['transforms']=[capture_transforms(model,bind,r,t) for r,t in zip(motion['rotations'],translation)]
    destination=Path(out) if out else take_dir
    if (destination/'physics_report.json').exists():
        raise FileExistsError('a physics bake already exists; use a fresh --out or create a new take')
    stage=destination/('.physics-'+uuid.uuid4().hex+'.partial')
    stage.mkdir(parents=True)
    cache=np.lib.format.open_memmap(stage/'physics_points.npy',mode='w+',dtype=np.float32,
                                  shape=(len(frames),len(arrays['surface_rest']),3))
    activations=np.zeros((len(frames),len(meta['muscles'])),np.float64)
    muscle_activations=np.zeros_like(activations)
    jaw=np.zeros((len(frames),3),np.float64)
    head_transforms=np.zeros((len(frames),4,4),np.float64)
    dental_transforms=np.zeros_like(head_transforms)
    reports=[];start=time.monotonic();profile=None
    definition=json.loads((rig_dir/'rig.json').read_text())
    evaluator=rigdef.Rig(definition)
    sensitivity_cache=SensitivityCache(sensitivity_reuse_frames) if sensitivity_reuse_frames and iterations else None
    contact_geometry=None;solver_sha256=None;actual_state_precision=None;actual_mechanics=None
    try:
        solver_path,solver_sha256=freeze_library(library_path)
        with cpu_threads(threads),Session(arrays,meta,path=solver_path,cancel=cancel,state_precision=state_precision,**mechanics_options) as state, \
                NativeTrialPool(state,inverse_workers,lazy=True) if iterations and inverse_workers>1 else nullcontext(None) as trial_pool:
            actual_state_precision=state.state_precision
            actual_mechanics=dict(contact_transport=state.contact_transport,jaw_mode=state.jaw_mode,
                                  jaw_track_m_rad=state.jaw_track.tolist(),jaw_inertia=state.jaw_inertia.tolist(),jaw_damping=state.jaw_damping.tolist(),
                                  inertial_repair_sweeps=state.inertial_repair_sweeps,volume_rescue_limit=state.volume_rescue_limit,external_muscle_cables=state.cable_count)
            neutral_report=state.advance(np.zeros(state.nm),1/30)
            neutral_rms=float(np.sqrt(np.mean(np.square(state.surface()-arrays['surface_rest'])))*1000)
            neutral_crossings=state.intersections()
            trial_geometry_filter=neutral_crossings==0
            previous_goal=None
            for i,(controls,target,profile) in enumerate(target_sequence(rig_dir,arrays,meta,frames,aux,motion=motion,speech_options=animation)):
                if cancel is not None and cancel.is_set():
                    raise RuntimeError('full-face simulation cancelled')
                if progress:
                    progress(.05+.8*i/len(frames),f'full-face inverse control {i+1}/{len(frames)}')
                prior=control_prior(meta,controls)
                extra_residual=None;contact_goals=[]
                if profile is not None:
                    if contact_geometry is None:
                        from .biomechanics_contacts import ContactGeometry
                        contact_geometry=ContactGeometry(arrays)
                    extra_residual,contact_goals=contact_geometry.objective(profile['values'][i])
                rigstate=evaluator.evaluate(controls)
                head=(motion['transforms'][i]['head'] if motion is not None else
                      rigstate['skin'][evaluator.jidx['head']])
                for part in meta['parts']:
                    if part['name'] not in ('eye_left','eye_right'):
                        continue
                    side='L' if part['name']=='eye_left' else 'R'
                    world=(motion['transforms'][i]['left_eye' if side=='L' else 'right_eye'] if motion is not None else
                           rigstate['skin'][evaluator.jidx['eye_'+side]])
                    matrix=np.linalg.inv(head)@world
                    surface_ids=np.arange(part['surface_start'],part['surface_start']+part['surface_count'])
                    volume_ids=np.unique(arrays['embed_ids'][surface_ids])
                    state.rigid_targets[volume_ids]=state.rest[volume_ids]@matrix[:3,:3].T+matrix[:3,3]
                dt=float(times[i]-times[i-1]) if i else 1/fps
                facial_request=any(abs(value)>1e-8 for name,value in controls.items()
                                   if not name.startswith(('head','eyeLook')))
                # The frame at t=0 is an equilibrated neutral initial condition.
                if (i==0 and not facial_request and not np.any(np.abs(target-arrays['surface_rest'])>.0001)
                        and np.allclose(state.rigid_targets,state.rest,atol=1e-10,rtol=0)):
                    chosen=np.zeros(state.nm)
                    report=dict(neutral_report,target_rms_before_mm=neutral_rms,target_rms_after_mm=neutral_rms,
                                evaluations=0,active_muscles=0,candidate_muscles=0,excited_muscles=0)
                else:
                    def validate_surface(surface):
                        from .biomechanics_native import GeometryStepError
                        exported=np.asarray(surface@head[:3,:3].T+head[:3,3],np.float32).astype(np.float64)
                        local_crossings=state.intersections(surface);crossings=state.intersections(exported)
                        triangles=surface_area_metrics(arrays['surface_rest'],exported,arrays['faces'])
                        if local_crossings or crossings or not triangles['passed']:
                            raise GeometryStepError('full-face trial violates surface geometry',dict(
                                head_local_surface_crossings=local_crossings,exported_surface_crossings=crossings,surface_triangles=triangles))
                    def target_met(surface):
                        regional=regional_target_metrics(arrays,meta,target,surface)
                        if not all(row['rms_mm']<=.75*row['limit_mm'] for row in regional.values()):return False
                        if np.sqrt(np.mean(((surface-target)/.001)**2))>.25:return False
                        if extra_residual is not None and np.max(np.abs(extra_residual(surface)),initial=0.)>.025:return False
                        if not surface_area_metrics(arrays['surface_rest'],surface,arrays['faces'])['passed']:return False
                        return state.intersections(surface)==0
                    chosen,report=inverse_step(state,target,prior,dt,iterations=iterations,sample_ids=inverse_samples(arrays,meta,target),
                                               previous=activations[i-1] if i else np.zeros(state.nm),cancel=cancel,
                                               trial_pool=trial_pool,
                                               sensitivities=sensitivity_cache,
                                               extra_residual=extra_residual,extra_weight=4.,probe_substeps=sensitivity_probe_substeps,
                                               stop=target_met,
                                               validate_trial=validate_surface if trial_geometry_filter else None,
                                               warm_start=activations[i-1] if i and np.array_equal(target,previous_goal) else None,
                                               progress=(lambda f,t:progress(.05+.8*(i+f)/len(frames),t)) if progress else None)
                # Cranial motion is applied exactly once after the head-local
                # coupled solve. Sampled USD geometry has no skin binding.
                surface=state.surface()
                report['articulator_contact_goals']=contact_goals
                report['regional_targets']=regional_target_metrics(arrays,meta,target,surface)
                report['head_local_surface_crossings']=state.intersections()
                ratios=volumes(state.positions,state.tets)/state.rest_volume
                report['volume_change_percent']=float(abs(np.sum(ratios*state.rest_volume)/np.sum(state.rest_volume)-1)*100)
                # Rigid pair distances use the complete dental surface, not a
                # skinned proxy. A fitted jaw is never allowed to stretch teeth.
                rigid_error=0.
                for part in meta['parts']:
                    if not part['name'].startswith('teeth_'):
                        continue
                    ids=np.arange(part['surface_start'],part['surface_start']+part['surface_count'])
                    r=arrays['surface_rest'][ids];p=surface[ids]
                    a,b=r-r.mean(0),p-p.mean(0)
                    u,_,vt=np.linalg.svd(a.T@b)
                    correction=np.diag([1.,1.,np.linalg.det(u@vt)])
                    rigid_error=max(rigid_error,float(np.linalg.norm(a@(u@correction@vt)-b,axis=1).max())*1000)
                report['dental_rigidity_max_mm']=rigid_error
                cache[i]=surface@head[:3,:3].T+head[:3,3]
                exported=np.asarray(cache[i],np.float64)
                report['proper_surface_crossings']=state.intersections(exported)
                report['surface_triangles']=surface_area_metrics(arrays['surface_rest'],exported,arrays['faces'])
                activations[i]=chosen;jaw[i]=state.q
                muscle_activations[i]=state.activation
                head_transforms[i]=head
                basis=state.transform(np.vstack((np.zeros(3),np.eye(3))))
                local=np.eye(4);local[:3,:3]=(basis[1:]-basis[0]).T;local[:3,3]=basis[0]
                dental_transforms[i]=head@local
                reports.append(report)
                previous_goal=target.copy()
            native_work=dict(native_steps=state.native_steps+sum(s.native_steps for s in trial_pool.sessions) if trial_pool is not None else state.native_steps,
                subdivision_retries=state.subdivision_retries+sum(s.subdivision_retries for s in trial_pool.sessions) if trial_pool is not None else state.subdivision_retries)
            native_cancellation=getattr(state.lib,'lightphysics_deformable_set_cancel_callback',None) is not None
        cache.flush()
        export_usd(stage/'physics.usda',arrays,meta,cache,times,fps)
        replay_error=verify_usd(stage/'physics.usda',meta,cache,times,fps)
        speech_checks=speech_contact_metrics(arrays,profile,cache,head_transforms)
        chunks=[]
        for index in range(0,len(frames),32):
            name=f'physics_points_{index//32:04d}.npz'
            np.savez_compressed(stage/name,positions=cache[index:index+32],timestamps=times[index:index+32],
                                jaw=jaw[index:index+32],excitations=activations[index:index+32],
                                activations=muscle_activations[index:index+32],
                                head_transforms=head_transforms[index:index+32],lower_dental_transforms=dental_transforms[index:index+32])
            chunks.append(dict(file=name,sha256=anatomy.digest(stage/name),start=index,count=min(32,len(frames)-index)))
        np.savez_compressed(stage/'physics_state.npz',times=times,jaw=jaw,excitations=activations,
                            activations=muscle_activations,muscle_names=np.asarray(meta['muscles']),
                            triangles=arrays['faces'],rest=arrays['surface_rest'],head_transforms=head_transforms,
                            lower_dental_transforms=dental_transforms,jaw_hinge=arrays['jaw_hinge'])
        if profile is not None:
            np.savez_compressed(stage/'articulation.npz',times=profile['times'],values=profile['values'])
            anatomy.write_json(stage/'articulation.json',{k:v for k,v in profile.items() if k not in ('times','values')})
        if review:
            review_movie(stage,arrays,meta,cache,fps,take_dir/'audio.wav',cancel=cancel)
        report=dict(schema='vhuman.biomechanics.v1',protocol='VHFULL1',backend='native_lightgeom_cpu',device='cpu',
                    geometry_only=True,anatomy_sha256=meta['arrays_sha256'],rig_sha256=meta['rig_sha256'],
                    anatomy_metadata_sha256=metadata_sha256,
                    muscle_atlas=meta.get('muscle_atlas','vhuman.authored_circular_fibers.v1'),
                    rig_package_sha256=meta['rig_package_sha256'],animation_sha256=anatomy.digest(take_dir/'animation.json'),
                    solver_sha256=solver_sha256,solver_library_frozen=True,
                    python_sources_at_start=python_sources_at_start,
                    skin_attachments=meta.get('skin_attachments',dict(model='isotropic_attachments')),
                    muscle_geometry=meta.get('muscle_geometry',dict(model='authored_spatial_bands')),
                    tongue_muscle_geometry=meta.get('tongue_muscle_geometry',dict(model='authored_spatial_bands')),
                    tongue_extrinsic_cables=meta.get('tongue_extrinsic_cables',dict(model='disabled')),
                    solver_settings=dict(inverse_iterations=iterations,inverse_workers=inverse_workers,
                        mechanics_solver=mechanics_solver,**actual_mechanics,
                        requested_state_precision=state_precision,state_transport_precision=actual_state_precision,exported_surface_precision='float32',
                        native_cancellation_checkpoints=native_cancellation,
                        trial_surface_geometry_filter=trial_geometry_filter,
                        sensitivity_reuse_frames=sensitivity_reuse_frames,
                        sensitivity_probe_substeps=sensitivity_probe_substeps,
                        inverse_stopping='regional RMS <= 0.75 of limits; all-surface RMS <= 0.25 mm; extra contact residual <= 0.025 mm; valid triangles; zero crossings',
                        step_hz=240,maximum_subdivisions=3,
                        material='neo_hookean_xpbd',actuation='implicit_tension_fibers',
                        correction_guard='continuous_signed_tet_volume',native_volume_floor=NATIVE_VOLUME_FLOOR),
                    contact_transport='common_volume_guard_fraction_on_embedded_corrections_and_reactions',
                    rigid_kinematics='rigid_part_substep_interpolation; adaptive retries bisect target motion',
                    native_work=native_work,
                    inverse_sample_policy=('parts+lip_features+eyelid_features+moving_frontal_skin' if 'fit_eye_features' in arrays else
                        'parts+lip_features+moving_frontal_skin') if 'fit_importance' in arrays else 'part_uniform_legacy',
                    alignment_sha256=anatomy.digest(take_dir/'align.json') if aux is not None else None,
                    capture_motion_sha256=anatomy.digest(take_dir/'motion.npz') if motion is not None else None,
                    frames=len(frames),fps=fps,parts=meta['parts'],
                    chunks=chunks,seconds=time.monotonic()-start,thread_budget=threads,
                    peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                    min_volume_ratio=min(r['min_volume_ratio'] for r in reports),
                    target_rms_mm=float(np.mean([r['target_rms_after_mm'] for r in reports])),per_frame=reports,
                    control_track='approximation; physics.usda/point caches are the evaluated geometry',
                    anatomical_accuracy='hidden anatomy, fiber fields and force scales are authored priors',
                    neutral_rms_mm=neutral_rms,neutral_proper_surface_crossings=neutral_crossings,
                    capture_fit_quality_gate=True if motion is not None else None,
                    cranial_motion_source='captured GNM joint transforms' if motion is not None else 'semantic rig controls',
                    usd_replay_max_error_mm=replay_error,speech_contact_checks=speech_checks,
                    quality_gate=False,review_required=True)
        report['numerically_valid']=report['min_volume_ratio']>=.0499
        report['geometry_checks']=dict(positive_volume=report['numerically_valid'],neutral_equilibrium=neutral_rms<=.1,
            contact_residual=all(r['contact_residual_mm']<=.05 for r in reports),
            proper_surface_crossings=neutral_crossings==0 and all(r['proper_surface_crossings']==0 for r in reports),
            surface_triangles=all(r['surface_triangles']['passed'] for r in reports),
            rigid_teeth=all(r['dental_rigidity_max_mm']<=.001 for r in reports),
            exact_usd_replay=replay_error<=.001,speech_closures=speech_checks['passed'],
            regional_target_tracking=all(all(v['passed'] for v in r['regional_targets'].values()) for r in reports),
            target_tracking=all(r['target_rms_after_mm']<=2 for r in reports))
        report['quality_gate']=all(report['geometry_checks'].values())
        report['status']='geometry_checks_passed' if report['quality_gate'] else 'rejected_geometry'
        report['simulation_real_time_factor']=report['seconds']/max(1/fps,float(times[-1]))
        anatomy.write_json(stage/'physics_report.json',report)
        del cache
        (stage/'physics_points.npy').unlink()
        for path in sorted(stage.iterdir(),key=lambda p:p.name=='physics_report.json'):
            if path.name.startswith('physics_') or path.name in ('physics.usda','articulation.json','articulation.npz'):
                path.replace(destination/path.name)
        stage.rmdir()
        if progress:
            progress(.99,'full-face geometry ready for review')
        return report
    except BaseException as exc:
        failure=dict(error=str(exc),anatomy_sha256=meta['arrays_sha256'],
                     anatomy_metadata_sha256=metadata_sha256,
                     completed_frames=len(reports),backend='cpu',solver_sha256=solver_sha256,
                     python_sources_at_start=python_sources_at_start,
                     requested_state_precision=state_precision,state_transport_precision=actual_state_precision,
                     mechanics_solver=mechanics_solver,mechanics=actual_mechanics)
        if hasattr(exc,'diagnostics'):failure['failed_step']=exc.diagnostics
        anatomy.write_json(stage/'failure.json',failure)
        raise RuntimeError(f'full-face simulation failed; diagnostics retained at {stage}: {exc}') from exc


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('rig',type=Path);ap.add_argument('take',type=Path)
    ap.add_argument('--out',type=Path);ap.add_argument('--package',type=Path)
    ap.add_argument('--threads',type=int,default=4);ap.add_argument('--iterations',type=int,default=2)
    ap.add_argument('--inverse-workers',type=int,default=1)
    ap.add_argument('--sensitivity-reuse-frames',type=int,default=0)
    ap.add_argument('--sensitivity-probe-substeps',type=int,default=0)
    ap.add_argument('--state-precision',choices=('auto','float32','float64'),default='auto')
    ap.add_argument('--mechanics-solver',choices=('partitioned','coupled','block'),default='partitioned',
                    help='experimental CPU coupled mechanics; block solves jaw attachments simultaneously')
    ap.add_argument('--no-review',action='store_true')
    args=ap.parse_args(argv)
    with anatomy.worker_cancellation() as cancel:
        report=simulate(args.rig,args.take,out=args.out,package=args.package,threads=args.threads,
                        iterations=args.iterations,inverse_workers=args.inverse_workers,
                        sensitivity_reuse_frames=args.sensitivity_reuse_frames,
                        sensitivity_probe_substeps=args.sensitivity_probe_substeps,state_precision=args.state_precision,
                        mechanics_solver=args.mechanics_solver,review=not args.no_review,cancel=cancel,
                        progress=lambda f,t:print(f'@progress {f:.3f} {t}',flush=True))
    print(json.dumps(report))


if __name__=='__main__':main()


def simulate_job(service,request,progress,cancel,*,python=None):
    from .biomechanics_trials import validate_workers,validate_iterations,validate_reuse_frames,validate_probe_substeps,validate_state_precision,validate_mechanics_solver
    validate_workers(request.get('inverse_workers',1),request.get('threads',4))
    validate_iterations(request.get('inverse_iterations',2))
    validate_reuse_frames(request.get('sensitivity_reuse_frames',0))
    validate_probe_substeps(request.get('sensitivity_probe_substeps',0))
    validate_state_precision(request.get('state_precision','auto'))
    validate_mechanics_solver(request.get('mechanics_solver','partitioned'),request.get('state_precision','auto'))
    head,take=request.get('head_id'),request.get('take_id')
    rig=service.rig_file(head,'rig.json').parent
    folder=service.take_file(head,take,'animation.json').parent
    arguments=[rig,folder,'--threads',request.get('threads',4),
               '--inverse-workers',request.get('inverse_workers',1),'--iterations',request.get('inverse_iterations',2),
               '--sensitivity-reuse-frames',request.get('sensitivity_reuse_frames',0),
               '--sensitivity-probe-substeps',request.get('sensitivity_probe_substeps',0),
               '--state-precision',request.get('state_precision','auto'),
               '--mechanics-solver',request.get('mechanics_solver','partitioned')]
    anatomy.run_worker('server.vhuman.rig.biomechanics',arguments,python=python,progress=progress,cancel=cancel)
    report=json.loads((folder/'physics_report.json').read_text())
    manifest_path=folder/'manifest.json'
    manifest=json.loads(manifest_path.read_text())
    manifest.update(motion_model='biomechanics',geometry='physics.usda',
                    geometry_review_required=report['review_required'],geometry_status=report['status'],
                    geometry_quality_gate=report['quality_gate'],rig_package_sha256=report['rig_package_sha256'],
                    anatomy_sha256=report['anatomy_sha256'])
    anatomy.write_json(manifest_path,manifest)
    return dict(head_id=head,take_id=take,report=report,
                usd_url=f'/v1/heads/{head}/rig/takes/{take}/physics.usda')
