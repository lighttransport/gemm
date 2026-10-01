"""Fetch immutable source and checksum-verified weights for native SenseVoice."""
from pathlib import Path
import hashlib
import subprocess
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
SOURCE_REVISION = 'a57c05bfe2a91b5e0cb0983479634eba3e28ede5'
LLAMA_REVISION = '803b7fcae893e9caaee3921779628fef83ac0965'
WEIGHT_REVISION = 'cebc2cdd171e895d783040dbd15f10f3a76f7151'
WEIGHT_SHA256 = '4ae45c94422de949b387e2e0fb10d7e14e4c42c69db30c3444ecc7d4b844b7c5'


def checkout(url, revision, path):
    if not path.exists():
        subprocess.run(['git', 'clone', '--filter=blob:none', '--no-checkout', url, str(path)], check=True)
        subprocess.run(['git', '-C', str(path), 'checkout', '--detach', revision], check=True)
    actual = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(path), 'status', '--porcelain', '--untracked-files=no'], text=True)
    if actual != revision or dirty:
        raise ValueError(f'Expected a clean checkout at {revision}: {path}')


def digest(path):
    with path.open('rb') as src:
        return hashlib.file_digest(src, 'sha256').hexdigest()


def main():
    root = ROOT / 'tmp/vhuman-emotion'
    root.mkdir(parents=True, exist_ok=True)
    checkout('https://github.com/modelscope/FunASR.git', SOURCE_REVISION, root / 'source')
    checkout('https://github.com/ggml-org/llama.cpp.git', LLAMA_REVISION, root / 'llama')
    weights = root / 'sensevoice-small-q8.gguf'
    if not weights.is_file() or digest(weights) != WEIGHT_SHA256:
        url = f'https://huggingface.co/FunAudioLLM/SenseVoiceSmall-GGUF/resolve/{WEIGHT_REVISION}/sensevoice-small-q8.gguf'
        partial = weights.with_suffix('.partial')
        try:
            with urllib.request.urlopen(url, timeout=60) as src, partial.open('wb') as dst:
                while data := src.read(1024 * 1024):
                    dst.write(data)
            if digest(partial) != WEIGHT_SHA256:
                raise ValueError('SenseVoice Q8 checkpoint checksum mismatch')
            partial.replace(weights)
        finally:
            partial.unlink(missing_ok=True)
    print('SenseVoice', weights, WEIGHT_SHA256, flush=True)


if __name__ == '__main__':
    main()
