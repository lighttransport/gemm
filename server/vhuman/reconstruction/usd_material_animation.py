"""Sample Blender shader sockets beside USD without retaining driver dependencies.

This is a Blender sidecar, not native USD shader animation. Values are exact
at captured frames; restored sockets interpolate linearly between samples.
"""
import json
import math
from pathlib import Path


FILENAME = 'blender_material_animation.json'
SCHEMA = 'vhuman.usd_blender_material_animation.v1'


def action_curves(animation):
    if animation and animation.action:
        for layer in animation.action.layers:
            for strip in layer.strips:
                for bag in strip.channelbags:
                    yield from bag.fcurves


def _set_frame(scene, frame):
    scene.frame_set(math.floor(frame), subframe=frame - math.floor(frame))


def _value(socket):
    value = socket.default_value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        result = float(value)
        values = [result]
    else:
        try:
            result = [float(x) for x in value]
        except (TypeError, ValueError):
            raise ValueError('only numeric material socket animation is supported') from None
        values = result
    if not values or len(values) > 16 or not all(math.isfinite(x) for x in values):
        raise ValueError('invalid animated material socket value')
    return result


def _socket(record):
    import bpy
    material = bpy.data.materials.get(record['material'])
    if material is None or not material.node_tree:
        raise ValueError('animated material missing')
    node = material.node_tree.nodes.get(record['node'])
    if node is None or node.bl_idname != record['node_type']:
        raise ValueError('animated material node differs')
    sockets = getattr(node, record['direction'])
    index = record['socket_index']
    if not 0 <= index < len(sockets):
        raise ValueError('animated material socket missing')
    socket = sockets[index]
    if socket.identifier != record['socket_identifier']:
        raise ValueError('animated material socket differs')
    return socket


def _read(directory):
    import numpy as np
    report = json.loads((Path(directory) / FILENAME).read_text())
    if report.get('schema') != SCHEMA:
        raise ValueError('unsupported material animation sidecar')
    frames = np.asarray(report['frames'], float)
    if (frames.ndim != 1 or not len(frames) or not np.isfinite(frames).all()
            or np.any(np.diff(frames) <= 0)):
        raise ValueError('invalid material animation frames')
    seen = set()
    for channel in report['channels']:
        key = (channel['material'], channel['node'], channel['direction'], channel['socket_index'])
        if key in seen or channel['direction'] not in ('inputs', 'outputs'):
            raise ValueError('duplicate or invalid material animation channel')
        seen.add(key)
        values = np.asarray(channel['values'], float)
        if (values.ndim not in (1, 2) or len(values) != len(frames)
                or not np.isfinite(values).all()
                or (values.ndim == 2 and not 1 <= values.shape[1] <= 16)):
            raise ValueError('invalid material animation values')
    return report


def capture(materials, directory, frames, *, scene=None, output_frames=None):
    import bpy
    scene = scene or bpy.context.scene
    frames = [float(frame) for frame in frames]
    if (not frames or not all(math.isfinite(x) for x in frames)
            or any(b <= a for a, b in zip(frames, frames[1:]))):
        raise ValueError('material sample frames must increase strictly')
    output_frames = frames if output_frames is None else [float(x) for x in output_frames]
    if (len(output_frames) != len(frames) or not all(math.isfinite(x) for x in output_frames)
            or any(b <= a for a, b in zip(output_frames, output_frames[1:]))):
        raise ValueError('material output frames must increase strictly and match samples')
    channels, sockets = [], []
    for material in materials:
        tree = material.node_tree
        if tree is None:
            continue
        animation = tree.animation_data
        curves = list(action_curves(animation)) + (list(animation.drivers) if animation else [])
        paths = {curve.data_path for curve in curves}
        found = set()
        for node in tree.nodes:
            for direction in ('inputs', 'outputs'):
                for index, socket in enumerate(getattr(node, direction)):
                    if not hasattr(socket, 'default_value'):
                        continue
                    path = socket.path_from_id('default_value')
                    if path not in paths:
                        continue
                    _value(socket)
                    channels.append(dict(material=material.name, node=node.name,
                        node_type=node.bl_idname, direction=direction, socket_index=index,
                        socket_identifier=socket.identifier, values=[]))
                    sockets.append(socket)
                    found.add(path)
        if paths - found:
            raise ValueError('unsupported animated material property: ' + ', '.join(sorted(paths - found)))
    previous = scene.frame_current + scene.frame_subframe
    try:
        for frame in frames:
            _set_frame(scene, frame)
            bpy.context.view_layer.update()
            for channel, socket in zip(channels, sockets):
                channel['values'].append(_value(socket))
    finally:
        _set_frame(scene, previous)
    report = dict(schema=SCHEMA, frames=output_frames, source_frames=frames, channels=channels,
        limitation='Blender socket samples with linear interpolation; no original driver dependencies or native USD shader-animation guarantee')
    (Path(directory) / FILENAME).write_text(json.dumps(report, indent=2) + '\n')
    return report


def restore(directory, *, scene=None):
    import bpy
    scene = scene or bpy.context.scene
    report = _read(directory)
    sockets = [_socket(channel) for channel in report['channels']]
    for channel, socket in zip(report['channels'], sockets):
        current, first = _value(socket), channel['values'][0]
        if (isinstance(current, list) != isinstance(first, list)
                or isinstance(current, list) and len(current) != len(first)):
            raise ValueError('animated material socket value shape differs')
    # This sidecar owns all animation on the captured material node trees.
    materials = {channel['material'] for channel in report['channels']}
    for name in materials:
        bpy.data.materials[name].node_tree.animation_data_clear()
    for channel, socket in zip(report['channels'], sockets):
        for frame, value in zip(report['frames'], channel['values']):
            socket.default_value = value
            socket.keyframe_insert('default_value', frame=frame)
    for name in materials:
        for curve in action_curves(bpy.data.materials[name].node_tree.animation_data):
            for point in curve.keyframe_points:
                point.interpolation = 'LINEAR'
    _set_frame(scene, report['frames'][0])
    return report


def verify(directory, *, scene=None):
    import bpy
    import numpy as np
    scene = scene or bpy.context.scene
    report = _read(directory)
    sockets = [_socket(channel) for channel in report['channels']]
    for channel, socket in zip(report['channels'], sockets):
        tree = bpy.data.materials[channel['material']].node_tree
        curves = [curve for curve in action_curves(tree.animation_data)
                  if curve.data_path == socket.path_from_id('default_value')]
        values = np.asarray(channel['values'])
        components = 1 if values.ndim == 1 else values.shape[1]
        if len(curves) != components or {c.array_index for c in curves} != set(range(components)):
            raise ValueError('restored material animation differs: missing component curves')
        for curve in curves:
            keys = np.array([point.co[:] for point in curve.keyframe_points])
            expected = values if values.ndim == 1 else values[:, curve.array_index]
            if (keys.shape != (len(report['frames']), 2)
                    or not np.allclose(keys[:, 0], report['frames'], atol=1e-7, rtol=1e-7)
                    or not np.allclose(keys[:, 1], expected, atol=1e-7, rtol=1e-7)):
                raise ValueError('restored material animation differs: sampled keys')
    for name in {channel['material'] for channel in report['channels']}:
        animation = bpy.data.materials[name].node_tree.animation_data
        if not animation or animation.drivers:
            raise ValueError('material animation still depends on drivers or is missing')
        for curve in action_curves(animation):
            if any(point.interpolation != 'LINEAR' for point in curve.keyframe_points):
                raise ValueError('material sample interpolation differs')
    previous = scene.frame_current + scene.frame_subframe
    maximum = 0.
    try:
        for i, frame in enumerate(report['frames']):
            _set_frame(scene, frame)
            bpy.context.view_layer.update()
            for channel, socket in zip(report['channels'], sockets):
                actual, expected = np.asarray(_value(socket)), np.asarray(channel['values'][i])
                if actual.shape != expected.shape or not np.allclose(actual, expected, atol=1e-7, rtol=1e-7):
                    raise ValueError('restored material animation differs')
                maximum = max(maximum, float(np.max(abs(actual - expected))))
    finally:
        _set_frame(scene, previous)
    return dict(passed=True, channels=len(sockets), frames=len(report['frames']),
                checks=len(sockets)*len(report['frames']), maximum_error=maximum,
                original_driver_dependencies_removed=True)
