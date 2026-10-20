"""Atomic, hash-verified offline tetrahedral volume reuse."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid


def _digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda:stream.read(1024*1024),b''):h.update(data)
    return h.hexdigest()


class MeshCache:
    def __init__(self,root):
        self.root=Path(root)

    def read(self,key,recipe):
        folder=self.root/key
        if not folder.exists():return None
        try:
            receipt=json.loads((folder/'receipt.json').read_text())
            output=folder/'volume.npz'
            if receipt['recipe']!=recipe or _digest(output)!=receipt['volume_sha256']:
                raise ValueError('receipt mismatch')
        except (OSError,ValueError,KeyError) as exc:
            raise ValueError(f'invalid tetrahedral cache entry; remove {folder} to remesh') from exc
        return output

    def publish(self,key,recipe,source):
        self.root.mkdir(parents=True,exist_ok=True)
        stage=self.root/('.'+key+'-'+uuid.uuid4().hex)
        stage.mkdir()
        try:
            output=stage/'volume.npz'
            with Path(source).open('rb') as src,output.open('xb') as dst:
                shutil.copyfileobj(src,dst);dst.flush();os.fsync(dst.fileno())
            (stage/'receipt.json').write_text(json.dumps(dict(recipe=recipe,volume_sha256=_digest(output)),allow_nan=False))
            try:stage.rename(self.root/key)
            except OSError:
                # Another complete writer can win. Verify its bytes/recipe;
                # never replace an existing entry with partially written data.
                if not (self.root/key).is_dir():raise
            return self.read(key,recipe)
        finally:
            if stage.exists():shutil.rmtree(stage)
