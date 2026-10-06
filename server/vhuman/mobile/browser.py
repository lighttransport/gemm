"""Build a self-contained WebGL2/WebAssembly avatar review directory.

Uses a locally extracted, pinned Three.js npm package; no CDN is needed at
runtime. Serve the output on localhost or HTTPS for package SHA256 verification.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
from .export import validate_package
from ..reconstruction.observations import sha256

ROOT=Path(__file__).resolve().parents[3]


def build(package,out,three,detail=None):
    package,out,three=map(lambda p:Path(p).resolve(),(package,out,three))
    validate_package(package)
    if json.loads((three/'package.json').read_text())['version']!='0.163.0':raise ValueError('expected Three.js 0.163.0')
    if out.exists() and any(out.iterdir()):raise ValueError('browser output must be empty')
    if detail:
        detail=Path(detail);record=json.loads((detail/'detail.json').read_text())
        if record['package_sha256']!=sha256(package/'avatar.json'):raise ValueError('detail package mismatch')
        for name,spec in record['files'].items():
            if Path(name).name!=name or sha256(detail/name)!=spec['sha256']:raise ValueError('detail checksum mismatch')
    out.mkdir(parents=True,exist_ok=True)
    (out/'compiler').mkdir()
    subprocess.run(['em++',str(Path(__file__).with_name('native.cpp')),'-std=c++17','-O3','-msimd128',
        '-sMODULARIZE=1','-sEXPORT_ES6=1','-sALLOW_MEMORY_GROWTH=1','-sINITIAL_MEMORY=134217728',
        '-sMAXIMUM_MEMORY=536870912','-sFILESYSTEM=1',
        '-sEXPORTED_FUNCTIONS=_vh_mobile_load,_vh_mobile_free,_vh_mobile_vertices,_vh_mobile_expressions,_vh_mobile_eval,_vh_mobile_joint_transform,_malloc,_free',
        '-sEXPORTED_RUNTIME_METHODS=FS,HEAPF32,HEAPU8','-o',str(out/'native.js')],
        check=True,env=dict(os.environ,TMPDIR=str(out/'compiler')))
    shutil.rmtree(out/'compiler')
    shutil.copytree(package,out/'avatar')
    if detail:shutil.copytree(detail,out/'detail')
    for source,target in [('build/three.module.js','three.module.js'),('examples/jsm/loaders/GLTFLoader.js','GLTFLoader.js'),
        ('examples/jsm/utils/BufferGeometryUtils.js','BufferGeometryUtils.js'),('LICENSE','THREE-LICENSE.txt')]:
        text=(three/source).read_text()
        text=text.replace("from 'three'","from './three.module.js'").replace("'../utils/BufferGeometryUtils.js'","'./BufferGeometryUtils.js'")
        (out/target).write_text(text)
    for name in ('vhuman_mobile.html','vhuman_mobile.js','vhuman_mobile_worker.js','vhuman_mobile_stream.js','vhuman_mobile_audio.js'):
        shutil.copyfile(ROOT/'web'/name,out/('index.html' if name.endswith('.html') else name))
    (out/'config.json').write_text(json.dumps(dict(schema='vhuman.browser.v1',package='avatar/',
        package_sha256=sha256(package/'avatar.json'),detail='detail/' if detail else None,
        detail_sha256=sha256(detail/'detail.json') if detail else None,three='0.163.0',runtime='WebGL2 + WASM SIMD')))
    return dict(out=str(out),serve=f'python -m http.server 8088 --bind 127.0.0.1 --directory {out}',url='http://127.0.0.1:8088/')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('package','out','three'):p.add_argument('--'+name,required=True)
    p.add_argument('--detail')
    print(json.dumps(build(**vars(p.parse_args())),indent=2))


if __name__=='__main__':main()
