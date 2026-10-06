"""Mobile WebSocket server for sample-aligned PCM and native GNM animation.

A provider is an async generator(request, epoch) yielding validated wire
records. Replay is a deterministic integration fixture; it does not synthesize
speech or invent visemes. Use speech.provider for the native TTS/student path.
"""
import argparse
import asyncio
import contextlib
import json
from pathlib import Path
import wave
import numpy as np
from . import wire
from .export import validate_package
from ..reconstruction.observations import sha256
from ..realtime.src.pipeline.protocol import AudioChunk


def replay_provider(wav, motion):
    with wave.open(str(wav), 'rb') as f:
        if (f.getnchannels(), f.getsampwidth(), f.getframerate()) != (1, 2, 24000):
            raise ValueError('replay requires 24kHz mono PCM16 WAV')
        if f.getnframes() > 24000 * 300: raise ValueError('replay exceeds five minutes')
        pcm = np.frombuffer(f.readframes(f.getnframes()), '<i2').astype(np.float32) / 32768
    with np.load(motion, allow_pickle=False) as z:
        samples = z['sample_positions']; expressions = z['expression']
        rotations = z['rotations']; translations = z['translation']
        geometry_hash = str(z['source_geometry_sha256'].item())
    if not len(samples) or samples[0] != 0 or np.any(np.diff(samples) <= 0) or samples[-1] >= len(pcm):
        raise ValueError('motion needs increasing audio sample positions starting at zero')
    if not len(samples) == len(expressions) == len(rotations) == len(translations):
        raise ValueError('motion arrays differ in length')
    # Validate the full clip before accepting clients.
    for s, x, r, t in zip(samples, expressions, rotations, translations): wire.pose(0, s, x, r, t)

    async def provider(request, epoch):
        if request['geometry_sha256'] != geometry_hash: raise ValueError('motion belongs to another identity')
        index = 0
        for sequence, start in enumerate(range(0, len(pcm), 1920)):
            end = min(start + 1920, len(pcm))
            # Send future motion before corresponding audio to allow interpolation.
            while index < len(samples) and (samples[index] < end or index == 0):
                yield wire.pose(epoch, samples[index], expressions[index], rotations[index], translations[index])
                index += 1
            yield wire.audio(AudioChunk(epoch, sequence, start, pcm[start:end]))
            await asyncio.sleep(len(pcm[start:end]) / 24000)
    return provider


def connection_handler(package, provider):
    manifest = validate_package(package)
    identity = manifest['source_geometry_sha256']; package_hash = sha256(Path(package) / 'avatar.json')

    async def client(socket):
        task = None; epoch = -1
        async def produce(request, current):
            try:
                order = wire.StreamOrder(); order.begin(current)
                async with contextlib.aclosing(provider(request, current)) as records:
                    async for record in records:
                        decoded = wire.decode(wire.encode(record))
                        if not order.accept(decoded): raise ValueError('provider returned stale data')
                        await socket.send(wire.encode(record))
                await socket.send(wire.encode(dict(type='end', epoch=current, samples=order.samples)))
            except asyncio.CancelledError: raise
            except Exception as e:
                await socket.send(wire.encode(dict(type='error', epoch=current, message=str(e))))
        try:
            first = json.loads(await asyncio.wait_for(socket.recv(), 10))
            if first != dict(type='hello', schema=wire.SCHEMA, package_sha256=package_hash):
                await socket.close(1008, 'avatar/protocol mismatch'); return
            await socket.send(wire.encode(dict(type='ready', schema=wire.SCHEMA, geometry_sha256=identity)))
            async for raw in socket:
                request = json.loads(raw)
                if not isinstance(request, dict) or request.get('type') not in ('speak', 'cancel'):
                    await socket.close(1008, 'invalid command'); return
                if task:
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError): await task
                    task = None
                epoch += 1
                await socket.send(wire.encode(dict(type='begin', epoch=epoch)))
                if request['type'] == 'speak':
                    if not isinstance(request.get('text'), str) or not 1 <= len(request['text']) <= 4096 or request.get('language') not in ('en', 'ja'):
                        await socket.close(1008, 'invalid speech request'); return
                    request['geometry_sha256'] = identity
                    task = asyncio.create_task(produce(request, epoch))
        finally:
            if task:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError): await task
    return client


async def serve(package, provider, host='127.0.0.1', port=8765):
    from websockets.asyncio.server import serve as websocket_serve
    async with websocket_serve(connection_handler(package, provider), host, port, max_size=wire.MAX_MESSAGE, max_queue=8):
        await asyncio.Future()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package', required=True); p.add_argument('--wav', required=True)
    p.add_argument('--motion', required=True); p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=8765)
    a = p.parse_args(); asyncio.run(serve(a.package, replay_provider(a.wav, a.motion), a.host, a.port))


if __name__ == '__main__': main()
