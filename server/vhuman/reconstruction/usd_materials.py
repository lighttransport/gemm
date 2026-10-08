"""Blender material sidecar for exact static shader restoration beside USD.

Run inside Blender. USD remains the geometry/material interchange; this sidecar
retains Blender node features that its importer does not roundtrip.
"""
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scalar(value):
    if isinstance(value, (str, bool, int, float)):
        return value
    try:
        values = list(value)
    except TypeError:
        return None
    if len(values) <= 16 and all(isinstance(v, (bool, int, float)) for v in values):
        return values
    return None


def color_ramp_record(ramp):
    return dict(color_mode=ramp.color_mode, interpolation=ramp.interpolation,
                hue_interpolation=ramp.hue_interpolation,
                elements=[dict(position=float(e.position), color=list(e.color))
                          for e in ramp.elements])


def capture(materials, out):
    import bpy
    out = Path(out)
    assets = out/'shader_assets'
    assets.mkdir(parents=True, exist_ok=True)
    image_records, records = {}, {}
    for material in materials:
        if not material.use_nodes:
            raise ValueError('node materials required')
        nodes = []
        for node in material.node_tree.nodes:
            if node.type == 'GROUP':
                raise ValueError('nested material node groups need explicit export support')
            props = {}
            for prop in node.bl_rna.properties:
                if prop.is_readonly or prop.type in ('POINTER', 'COLLECTION'):
                    continue
                if prop.identifier in ('location', 'dimensions', 'width', 'height', 'select'):
                    continue
                value = scalar(getattr(node, prop.identifier))
                if value is not None:
                    props[prop.identifier] = value
            entry = dict(name=node.name, type=node.bl_idname, properties=props,
                         inputs={str(i):scalar(s.default_value) for i, s in enumerate(node.inputs)
                                 if hasattr(s, 'default_value') and scalar(s.default_value) is not None})
            if node.type == 'VALTORGB':
                entry['color_ramp'] = color_ramp_record(node.color_ramp)
            if node.type == 'TEX_IMAGE' and node.image:
                image = node.image
                if image.name not in image_records:
                    suffix = '.exr' if image.file_format == 'OPEN_EXR' else Path(image.filepath).suffix or '.png'
                    path = assets/(f'{len(image_records):03d}'+suffix)
                    if image.packed_file:
                        path.write_bytes(bytes(image.packed_file.data))
                    else:
                        path.write_bytes(Path(bpy.path.abspath(image.filepath)).read_bytes())
                    image_records[image.name] = dict(path=str(path.relative_to(out)), sha256=digest(path),
                        colorspace=image.colorspace_settings.name, alpha_mode=image.alpha_mode)
                entry['image'] = image.name
            nodes.append(entry)
        records[material.name] = dict(nodes=nodes,
            links=[dict(source=l.from_node.name, source_socket=list(l.from_node.outputs).index(l.from_socket),
                        target=l.to_node.name, target_socket=list(l.to_node.inputs).index(l.to_socket))
                   for l in material.node_tree.links],
            settings={name:scalar(getattr(material, name)) for name in
                      ('diffuse_color', 'displacement_method', 'surface_render_method', 'use_backface_culling')
                      if hasattr(material, name)})
    report = dict(schema='vhuman.usd_blender_materials.v1', materials=records, images=image_records,
                  limitation='Static node values only; animation drivers and node groups are not exported')
    (out/'blender_materials.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


def restore(directory):
    import bpy
    root = Path(directory).resolve()
    report = json.loads((root/'blender_materials.json').read_text())
    if report.get('schema') != 'vhuman.usd_blender_materials.v1':
        raise ValueError('unsupported material sidecar')
    images = {}
    for name, record in report['images'].items():
        path = (root/record['path']).resolve()
        if not path.is_relative_to(root) or digest(path) != record['sha256']:
            raise ValueError('material image path/hash mismatch')
        image = bpy.data.images.load(str(path), check_existing=False)
        image.colorspace_settings.name = record['colorspace']
        image.alpha_mode = record['alpha_mode']
        image.pack()
        images[name] = image
    for name, record in report['materials'].items():
        material = bpy.data.materials.get(name)
        if material is None:
            raise ValueError('USD material binding missing: '+name)
        material.use_nodes = True
        material.node_tree.nodes.clear()
        for entry in record['nodes']:
            node = material.node_tree.nodes.new(entry['type'])
            for key, value in entry['properties'].items():
                setattr(node, key, value)
            node.name = entry['name']
            if 'color_ramp' in entry:
                spec, ramp = entry['color_ramp'], node.color_ramp
                for key in ('color_mode', 'interpolation', 'hue_interpolation'):
                    setattr(ramp, key, spec[key])
                while len(ramp.elements) > 1:
                    ramp.elements.remove(ramp.elements[-1])
                for i, stop in enumerate(spec['elements']):
                    element = ramp.elements[0] if i == 0 else ramp.elements.new(stop['position'])
                    element.position = stop['position']
                    element.color = stop['color']
            if 'image' in entry:
                node.image = images[entry['image']]
            for index, value in entry['inputs'].items():
                node.inputs[int(index)].default_value = value
        for link in record['links']:
            nodes = material.node_tree.nodes
            material.node_tree.links.new(nodes[link['source']].outputs[link['source_socket']],
                                         nodes[link['target']].inputs[link['target_socket']])
        for key, value in record['settings'].items():
            setattr(material, key, value)
    return report


def verify(directory):
    """Check restored node values, links and texture contents, not only names."""
    import bpy
    import numpy as np
    root = Path(directory).resolve()
    report = json.loads((root/'blender_materials.json').read_text())
    count = 0
    for name, record in report['materials'].items():
        material = bpy.data.materials.get(name)
        if material is None or len(material.node_tree.nodes) != len(record['nodes']):
            raise ValueError('restored material node count differs: '+name)
        nodes = material.node_tree.nodes
        for entry in record['nodes']:
            node = nodes.get(entry['name'])
            if node is None or node.bl_idname != entry['type']:
                raise ValueError('restored material node differs')
            for key, expected in entry['properties'].items():
                actual = scalar(getattr(node, key))
                equal = actual == expected if isinstance(expected, (str, bool)) else np.allclose(actual, expected, atol=1e-7, rtol=1e-7)
                if not equal:
                    raise ValueError('restored node property differs: '+key)
            if 'color_ramp' in entry:
                actual, expected = color_ramp_record(node.color_ramp), entry['color_ramp']
                if (any(actual[key] != expected[key] for key in
                        ('color_mode', 'interpolation', 'hue_interpolation'))
                        or len(actual['elements']) != len(expected['elements'])
                        or any(not np.allclose([a['position'], *a['color']],
                                               [b['position'], *b['color']], atol=1e-7, rtol=1e-7)
                               for a, b in zip(actual['elements'], expected['elements']))):
                    raise ValueError('restored color ramp differs')
            for index, expected in entry['inputs'].items():
                actual = scalar(node.inputs[int(index)].default_value)
                equal = actual == expected if isinstance(expected, (str, bool)) else np.allclose(actual, expected, atol=1e-7, rtol=1e-7)
                if not equal:
                    raise ValueError('restored socket value differs')
            if 'image' in entry:
                image = node.image
                target = report['images'][entry['image']]
                if (not image or not image.packed_file
                        or hashlib.sha256(bytes(image.packed_file.data)).hexdigest() != target['sha256']
                        or image.colorspace_settings.name != target['colorspace']
                        or image.alpha_mode != target['alpha_mode']):
                    raise ValueError('restored image differs')
            count += 1
        actual_links = {(l.from_node.name, list(l.from_node.outputs).index(l.from_socket),
                         l.to_node.name, list(l.to_node.inputs).index(l.to_socket))
                        for l in material.node_tree.links}
        expected_links = {(l['source'], l['source_socket'], l['target'], l['target_socket']) for l in record['links']}
        if actual_links != expected_links:
            raise ValueError('restored material links differ')
        for key, expected in record['settings'].items():
            actual = scalar(getattr(material, key))
            equal = actual == expected if isinstance(expected, (str, bool)) else np.allclose(actual, expected)
            if not equal:
                raise ValueError('restored material settings differ')
    return dict(materials=len(report['materials']), nodes=count, images=len(report['images']),
                settings_links_and_packed_images_verified=True)
