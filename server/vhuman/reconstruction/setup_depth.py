"""Explicit Small-only model setup. Caller chooses immutable revisions and hash.

python -m server.vhuman.reconstruction.setup_depth --code-revision <commit>
  --weight-revision <commit> --weight-sha256 <sha> [--out tmp/...]
"""
import argparse
import json
import re
import subprocess
import urllib.request
from pathlib import Path
from .observations import sha256


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--code-revision',default='a561b849ebae10a6f5ef49e26c83cbbcd36c71bf')
    ap.add_argument('--weight-revision',default='03876f8651c73a60fe4c2c48294e09fcb6838fcf')
    ap.add_argument('--weight-sha256',default='715fade13be8f229f8a70cc02066f656f2423a59effd0579197bbf57860e1378')
    ap.add_argument('--out',default='tmp/vhuman-rig/models/depth-anything-v2-small')
    a = ap.parse_args()
    for revision in (a.code_revision,a.weight_revision):
        if not re.fullmatch('[0-9a-f]{40}',revision):
            ap.error('revisions must be immutable 40-digit Git hashes')
    if not re.fullmatch('[0-9a-f]{64}',a.weight_sha256):
        ap.error('weight hash must be SHA256')
    out = Path(a.out)
    out.mkdir(parents=True,exist_ok=True)
    source = out/'source'
    if not source.exists():
        subprocess.run(['git','clone','https://github.com/DepthAnything/Depth-Anything-V2.git',str(source)],check=True)
    subprocess.run(['git','-C',str(source),'checkout','--detach',a.code_revision],check=True)
    url = f'https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/{a.weight_revision}/depth_anything_v2_vits.pth'
    weights = out/'depth_anything_v2_vits.pth'
    partial = out/'weights.partial'
    try:
        urllib.request.urlretrieve(url,partial)
        if sha256(partial)!=a.weight_sha256:
            raise ValueError('Small checkpoint hash mismatch')
        partial.replace(weights)
    finally:
        partial.unlink(missing_ok=True)
    (out/'installation.json').write_text(json.dumps(dict(model='Depth-Anything-V2-Small',license='Apache-2.0',
        code_revision=a.code_revision,weight_revision=a.weight_revision,weights_sha256=a.weight_sha256,source=url),indent=2))

if __name__=='__main__':
    main()
