"""Research media/weight destinations must stay out of tracked source."""
from pathlib import Path


def artifact_path(path):
    path=Path(path)
    repo=Path(__file__).resolve().parents[3]
    resolved=path.resolve()
    if resolved.is_relative_to(repo) and not resolved.is_relative_to(repo/'tmp'):
        raise ValueError('generated media/weights must be outside the repository or in ignored tmp/')
    return path
