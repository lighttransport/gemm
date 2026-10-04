"""Lossless, bounded reads of native F32 diagnostic captures."""
import gzip
from pathlib import Path


def read_f32(directory, name, count):
    import numpy as np
    directory = Path(directory)
    raw = directory / (name + ".f32")
    count = int(count)
    if count <= 0:
        raise ValueError("capture element count must be positive")
    expected = count * 4
    if raw.is_file():
        if raw.stat().st_size != expected:
            raise ValueError("native capture byte count mismatch")
        return np.fromfile(raw, dtype="<f4", count=count)
    with gzip.open(directory / (name + ".f32.gz"), "rb") as stream:
        data = stream.read(expected + 1)
    if len(data) != expected:
        raise ValueError("compressed native capture byte count mismatch")
    return np.frombuffer(data, dtype="<f4").copy()


def compress(directory, cancelled=None):
    """Compress completed native captures atomically, one bounded chunk at a time."""
    directory = Path(directory)
    for raw in sorted(directory.glob("*.f32")):
        if cancelled and cancelled():
            raise InterruptedError("capture compression cancelled")
        target = raw.with_suffix(raw.suffix + ".gz")
        stage = target.with_name(target.name + ".partial")
        if target.exists() or stage.exists():
            raise FileExistsError("compressed capture already exists")
        try:
            with raw.open("rb") as source, stage.open("xb") as destination:
                with gzip.GzipFile(filename="", mode="wb", compresslevel=1,
                                   fileobj=destination, mtime=0) as packed:
                    while chunk := source.read(1 << 20):
                        if cancelled and cancelled():
                            raise InterruptedError("capture compression cancelled")
                        packed.write(chunk)
            stage.replace(target)
            raw.unlink()
        finally:
            stage.unlink(missing_ok=True)


def write_npy(path, value, compressed=False):
    """Store independent arrays losslessly; return the path bound by a receipt."""
    import numpy as np
    path = Path(path)
    if not compressed:
        np.save(path, value, allow_pickle=False)
        return path
    path = path.with_suffix(path.suffix + ".gz")
    with path.open("wb") as output:
        with gzip.GzipFile(filename="", fileobj=output, mode="wb",
                           compresslevel=1, mtime=0) as stream:
            np.save(stream, value, allow_pickle=False)
    return path


def read_npy(path):
    import numpy as np
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as stream:
            return np.load(stream, allow_pickle=False)
    return np.load(path, allow_pickle=False)
