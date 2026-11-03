# Smoke 00: editor + python + MetaHuman subsystem exist; report engine/RHI/plugin state. Runs inside UE.
import os
import sys

import unreal

import mh_common as mh

OUT = os.path.join(mh.OUT_ROOT, 'smoke', '00_env.json')


def main():
    info = dict(engine_version=unreal.SystemLibrary.get_engine_version(), python=sys.version,
                argv=sys.argv, cwd=os.getcwd())
    sub = mh.subsystem()
    info['metahuman_subsystem'] = sub is not None and sub.get_class().get_name()
    mh_plugin = unreal.Paths.combine([unreal.Paths.engine_plugins_dir(), 'MetaHuman', 'MetaHumanCharacter'])
    info['metahuman_plugin_dir'] = mh_plugin
    optional = unreal.Paths.combine([mh_plugin, 'Content', 'Optional'])
    info['optional_content_present'] = unreal.Paths.directory_exists(optional)
    info['texture_synthesis_dir_present'] = unreal.Paths.directory_exists(unreal.Paths.combine([optional, 'TextureSynthesis']))
    info['classes'] = {n: hasattr(unreal, n) for n in ('MetaHumanCharacter', 'MetaHumanCharacterFactoryNew', 'MetaHumanCharacterSkinSettings',
                                                        'MetaHumanCharacterEyesSettings', 'MetaHumanCharacterExportBlueprintLibrary',
                                                        'MetaHumanGeometryExportParams', 'GLTFExporter', 'MoviePipelineQueueSubsystem',
                                                        'RenderingLibrary', 'SceneCapture2D')}
    info['subsystem_methods'] = sorted(m for m in dir(sub) if not m.startswith('_'))[:200]
    info['editor_world'] = str(mh.editor_world().get_name())
    mh.log(f'engine {info["engine_version"]} subsystem {info["metahuman_subsystem"]} optional {info["optional_content_present"]}')
    return info


mh.guarded(main, OUT)
