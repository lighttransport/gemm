"""ROCm native Qwen TTS + causal motion provider for the mobile stream.

Requires an explicitly geometry-matched semantic-to-GNM map. This adapter
refuses to silently substitute another actor's control map or an untrained
student. English/Japanese quality still requires a held-out utterance suite.
"""
import argparse
import asyncio
import json
from pathlib import Path
import queue
import tempfile
import numpy as np
from . import wire
from .export import validate_package
from .stream import serve
from ..realtime.src.tts.native import NativeTTS
from ..realtime.src.animation.causal import MotionAdapter
from ..realtime.src.pipeline.protocol import AudioChunk


class NativeMapping:
    def __init__(self, path, package):
        manifest = validate_package(package)
        control = json.loads((Path(package)/'controls.json').read_text())
        spec = json.loads(Path(path).read_text())
        if (spec.get('schema') != 'vhuman.gnm_expression_map.v1' or
                spec.get('source_geometry_sha256') != manifest['source_geometry_sha256'] or
                spec.get('coefficient_names') != control['names']):
            raise ValueError('speech mapping must match avatar geometry and native coefficient ordering')
        self.names = spec['controls']; self.matrix = np.asarray(spec['coefficients'], np.float32)
        self.reference = np.asarray(control['reference'], np.float32)
        if not self.names or len(set(self.names)) != len(self.names) or self.matrix.shape != (383, len(self.names)) or not np.isfinite(self.matrix).all():
            raise ValueError('invalid semantic-to-native matrix')

    def evaluate(self, values, names):
        if tuple(names) != tuple(self.names): raise ValueError('student/mapping semantic control ordering differs')
        weights = np.asarray(values, np.float32)
        if weights.shape != (len(self.names),) or not np.isfinite(weights).all(): raise ValueError('invalid student controls')
        return np.clip(self.reference + self.matrix @ np.clip(weights, 0, 1), -3, 3)


def provider(args):
    mapping = NativeMapping(args.mapping, args.package)
    # One GPU synthesis job at a time on the 16GB server. The socket handler
    # joins cancellation before beginning a new epoch.
    lock = asyncio.Lock()
    async def produce(request, epoch):
        async with lock:
            Path(args.work).mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix='speech-', dir=args.work) as work:
                source = adapter = None
                try:
                    adapter = MotionAdapter(args.adapter, args.revision, allow_diagnostic=args.diagnostic)
                    adapter.reset(epoch)
                    if tuple(adapter.model.names) != tuple(mapping.names): raise ValueError('student/mapping control mismatch')
                    source = NativeTTS(args.runner, args.model, args.revision, request['text'], work,
                        epoch=epoch, speaker=args.speaker_ja if request['language']=='ja' else args.speaker_en,
                        max_frames=args.max_frames, threads=args.threads, text_feed=adapter.text_feed,
                        backend=args.backend, language='Japanese' if request['language']=='ja' else 'English')
                    pending = None; feature_end = 0; sequence = 0; sent = 0
                    deadline = asyncio.get_running_loop().time()
                    while True:
                        source.check()
                        # Bound lookahead to 400ms; 100Hz motion never grows an
                        # unbounded queue while PCM or the client is stalled.
                        while feature_end < sent + 9600:
                            try: features = source.features.get_nowait()
                            except queue.Empty: break
                            for frame in adapter.push(features):
                                yield wire.pose(epoch, frame.sample_position, mapping.evaluate(frame.controls, adapter.model.names))
                            feature_end = features.sample_start + features.sample_count
                        if pending is None:
                            try: pending = source.audio.get_nowait()
                            except queue.Empty: pass
                        if pending is not None and pending.sample_start+len(pending.pcm) <= feature_end:
                            if pending.sample_start != sent: raise ValueError('native PCM discontinuity')
                            for start in range(0, len(pending.pcm), 1920):
                                pcm = pending.pcm[start:start+1920]
                                yield wire.audio(AudioChunk(epoch, sequence, sent, pcm))
                                sent += len(pcm); sequence += 1; deadline += len(pcm)/24000
                                await asyncio.sleep(max(0, deadline-asyncio.get_running_loop().time()))
                            pending = None
                        if source.done_audio.is_set() and source.done_features.is_set() and source.audio.empty() and source.features.empty():
                            if pending is not None: raise ValueError('PCM extends beyond native motion features')
                            break
                        await asyncio.sleep(.002)
                finally:
                    # NativeTTS.close kills the process and joins both pipe
                    # readers before releasing the temporary files and GPU slot.
                    if source is not None: source.close()
                    if adapter is not None: adapter.close()
    return produce


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('package', 'mapping', 'adapter', 'runner', 'model', 'revision'): p.add_argument('--'+name, required=True)
    p.add_argument('--backend', choices=['rocm', 'cpu', 'cuda'], default='rocm')
    p.add_argument('--work', default='tmp/vhuman-mobile/speech'); p.add_argument('--threads', type=int, default=8)
    p.add_argument('--speaker-en', default='Ryan'); p.add_argument('--speaker-ja', default='Ono_Anna')
    p.add_argument('--max-frames', type=int, default=512); p.add_argument('--diagnostic', action='store_true')
    p.add_argument('--host', default='127.0.0.1'); p.add_argument('--port', type=int, default=8765)
    a = p.parse_args()
    if not 1 <= a.threads <= 128 or not 1 <= a.max_frames <= 4096: p.error('invalid native inference limits')
    async def run(): await serve(a.package, provider(a), a.host, a.port)
    asyncio.run(run())


if __name__ == '__main__': main()
