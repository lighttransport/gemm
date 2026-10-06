"""Mobile geometry parity, package integrity and transport failure gates."""
import base64
import asyncio
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from .mobile import wire
from .mobile.export import hair_cards, validate_package
from .mobile.native import Native, build, write_model
from .reconstruction.observations import sha256
from .realtime.src.pipeline.protocol import AudioChunk

WORK = Path(__file__).resolve().parents[2] / 'tmp/vhuman-mobile-tests'


class WireTests(unittest.TestCase):
    def test_pcm_roundtrip_and_order(self):
        x = AudioChunk(0, 0, 0, np.linspace(-1, 1, 1920, dtype=np.float32))
        result = wire.decode(wire.encode(wire.audio(x)))
        np.testing.assert_array_equal(x.pcm, result.pcm)
        order = wire.StreamOrder(); order.begin(0); self.assertTrue(order.accept(result))
        with self.assertRaises(ValueError): order.accept(result)
        order.begin(1); self.assertFalse(order.accept(result))
        with self.assertRaises(ValueError): order.begin(1)
        with self.assertRaises(ValueError): order.accept(AudioChunk(2, 0, 0, np.ones(1)))

    def test_motion_shapes_bounds_and_monotonic_clock(self):
        result = wire.decode(wire.encode(wire.pose(0, 0, np.zeros(383))))
        order = wire.StreamOrder(); order.begin(0); order.accept(result)
        with self.assertRaises(ValueError): order.accept(result)
        for bad in (np.zeros(60), np.full(383, np.nan), np.full(383, 3.1)):
            with self.assertRaises(ValueError): wire.pose(0, 0, bad)
        with self.assertRaises(ValueError): wire.pose(True, 0, np.zeros(383))

    def test_malformed_or_excessive_audio_rejected(self):
        record = wire.audio(AudioChunk(0, 0, 0, np.ones(1)))
        for pcm in ('!', base64.b64encode(b'abc').decode(), base64.b64encode(np.array([np.nan], '<f4')).decode()):
            with self.assertRaises(ValueError): wire.decode(wire.encode(dict(record, pcm=pcm)))
        with self.assertRaises(ValueError): wire.audio(AudioChunk(0, 0, 0, np.zeros(4801)))
        with self.assertRaises(ValueError): wire.audio(AudioChunk(0, 0, 0, np.array([1.1])))
        with self.assertRaises(ValueError): wire.decode(' ' * (wire.MAX_MESSAGE + 1))


class PackageTests(unittest.TestCase):
    def test_hash_inventory_and_traversal(self):
        WORK.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=WORK) as temp:
            root = Path(temp); files = {}
            for name in ('gnm.bin', 'avatar.glb', 'streams.bin', 'bindings.bin', 'controls.json'):
                (root/name).write_bytes(b'fixture')
                files[name] = dict(sha256=sha256(root/name), bytes=7)
            manifest = dict(schema='vhuman.mobile_avatar.v1', files=files)
            def write(): (root/'avatar.json').write_text(json.dumps(manifest))
            write(); validate_package(root)
            (root/'gnm.bin').write_bytes(b'changed')
            with self.assertRaises(ValueError): validate_package(root)
            (root/'gnm.bin').write_bytes(b'fixture')
            files['../escape'] = dict(sha256='0'*64, bytes=7); write()
            with self.assertRaises(ValueError): validate_package(root)
            del files['../escape']; del files['gnm.bin']; write()
            with self.assertRaises(ValueError): validate_package(root)

    def test_hair_ribbons_have_deterministic_root_layout_and_budget(self):
        rng = np.random.default_rng(7); paths = np.cumsum(rng.normal(0, .001, (1500, 8, 3)), axis=1)
        p, tri, uv, selected = hair_cards(paths)
        self.assertEqual(len(tri), 4096); self.assertEqual(p.shape, (6144, 3))
        self.assertEqual(uv.shape, (4096, 3, 2)); self.assertTrue(np.isfinite(p).all())
        np.testing.assert_allclose(p.reshape(-1, 3, 2, 3)[:, 0].mean(1), paths[selected, 0])
        self.assertEqual(len(hair_cards(np.zeros((0, 8, 3)))[0]), 0)


class SocketTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        try:
            from websockets.asyncio.server import serve
            from websockets.asyncio.client import connect
        except ImportError: self.skipTest('install mobile/requirements.txt')
        self.serve, self.connect = serve, connect
        WORK.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=WORK); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        files = {}
        for name in ('gnm.bin', 'avatar.glb', 'streams.bin', 'bindings.bin', 'controls.json'):
            (self.root/name).write_bytes(b'fixture')
            files[name] = dict(bytes=7, sha256=sha256(self.root/name))
        (self.root/'avatar.json').write_text(json.dumps(dict(schema='vhuman.mobile_avatar.v1', files=files, source_geometry_sha256='a'*64)))

    async def receive(self, socket):
        return json.loads(await asyncio.wait_for(socket.recv(), 3))

    async def test_identity_handshake_and_cancel_joins_old_producer(self):
        from .mobile.stream import connection_handler
        joined = []
        async def provider(request, epoch):
            try:
                self.assertEqual(request['geometry_sha256'], 'a'*64)
                yield wire.pose(epoch, 0, np.zeros(383))
                yield wire.audio(AudioChunk(epoch, 0, 0, np.zeros(1920)))
                await asyncio.Future()
            finally: joined.append(epoch)
        async with self.serve(connection_handler(self.root, provider), '127.0.0.1', 0) as server:
            url = 'ws://127.0.0.1:'+str(server.sockets[0].getsockname()[1])
            async with self.connect(url) as socket:
                await socket.send(wire.encode(dict(type='hello', schema=wire.SCHEMA, package_sha256=sha256(self.root/'avatar.json'))))
                self.assertEqual((await self.receive(socket))['type'], 'ready')
                for epoch in (0, 2):
                    await socket.send(wire.encode(dict(type='speak', text='こんにちは', language='ja')))
                    self.assertEqual(await self.receive(socket), dict(type='begin', epoch=epoch))
                    self.assertEqual((await self.receive(socket))['type'], 'motion')
                    self.assertEqual((await self.receive(socket))['type'], 'audio')
                    await socket.send(wire.encode(dict(type='cancel')))
                    self.assertEqual(await self.receive(socket), dict(type='begin', epoch=epoch+1))
                    self.assertIn(epoch, joined)
            async with self.connect(url) as socket:
                await socket.send(wire.encode(dict(type='hello', schema=wire.SCHEMA, package_sha256='wrong')))
                from websockets.exceptions import ConnectionClosedError
                with self.assertRaises(ConnectionClosedError): await socket.recv()

    async def test_provider_audio_gap_fails_before_transmission(self):
        from .mobile.stream import connection_handler
        async def provider(request, epoch):
            yield wire.audio(AudioChunk(epoch, 1, 0, np.zeros(1920)))
        async with self.serve(connection_handler(self.root, provider), '127.0.0.1', 0) as server:
            async with self.connect('ws://127.0.0.1:'+str(server.sockets[0].getsockname()[1])) as socket:
                await socket.send(wire.encode(dict(type='hello', schema=wire.SCHEMA, package_sha256=sha256(self.root/'avatar.json'))))
                await self.receive(socket)
                await socket.send(wire.encode(dict(type='speak', text='hello', language='en')))
                await self.receive(socket)
                result = await self.receive(socket)
                self.assertEqual(result['type'], 'error'); self.assertIn('discontinuity', result['message'])


class NativeMobileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from .face_assets import asset_path
        from .rig.gnm_model import GNMModel
        if not asset_path('gnm').is_file(): raise unittest.SkipTest('GNM weights unavailable')
        WORK.mkdir(parents=True, exist_ok=True)
        cls.temp = tempfile.TemporaryDirectory(dir=WORK); cls.root = Path(cls.temp.name)
        cls.model = GNMModel(); cls.library = build(cls.root)
        rng = np.random.default_rng(41); cls.beta = rng.normal(0, .1, 253)
        rest, _ = cls.model.evaluate(cls.beta)
        cls.residual = rng.normal(0, .001, rest.shape)
        from scipy.spatial.transform import Rotation
        cls.rotation = Rotation.from_rotvec([.15, -.2, .05]).as_matrix()
        cls.scale = 1.1; cls.offset = np.array([.03, .02, -.01])
        cls.geometry = dict(gnm_identity=cls.beta, full_neutral=cls.scale*(rest+cls.residual)@cls.rotation.T+cls.offset,
            scale=np.array(cls.scale), rotation=cls.rotation)
        write_model(cls.root/'gnm.bin', cls.model, cls.geometry)

    @classmethod
    def tearDownClass(cls): cls.temp.cleanup()

    def test_expression_lbs_residual_parity_and_joint_affine(self):
        rng = np.random.default_rng(17)
        with Native(self.library, self.root/'gnm.bin') as native:
            with self.assertRaises(ValueError): native.joint_transform(0)
            for _ in range(5):
                x = rng.normal(0, .12, 383); r = rng.normal(0, .2, (4, 3)); t = rng.normal(0, .01, 3)
                # The writer bakes any constant residual offset into the origin;
                # reconstruct the exact bind correction represented in the file.
                rest, _ = self.model.evaluate(self.beta)
                origin = np.median(self.geometry['full_neutral']-self.scale*rest@self.rotation.T, axis=0)
                bind = (self.geometry['full_neutral']-origin)@self.rotation/self.scale-rest
                ref, _ = self.model.evaluate(self.beta, x, r, t, bind_residual=bind)
                ref = self.scale*ref@self.rotation.T+origin
                got = native.evaluate(x, r, t)
                error = np.linalg.norm(got-ref, axis=1)*1000
                self.assertLess(np.percentile(error, 95), .25)
            # Pure root rotation must move both the face and attached optics.
            r[:] = 0; r[0, 1] = .4
            neutral = native.evaluate(np.zeros(383)); moved = native.evaluate(np.zeros(383), r)
            matrix, offset = native.joint_transform(1)
            np.testing.assert_allclose(neutral@matrix.T+offset, moved, atol=1e-6)

    def test_invalid_model_and_pose_rejected(self):
        (self.root/'bad.bin').write_bytes(b'VHGNM001'+b'\xff'*16)
        with self.assertRaises(ValueError): Native(self.library, self.root/'bad.bin')
        with Native(self.library, self.root/'gnm.bin') as native:
            with self.assertRaises(ValueError): native.evaluate(np.full(383, np.nan))
            with self.assertRaises(ValueError): native.evaluate(np.zeros(383), np.zeros(12))
            with self.assertRaises(ValueError): native.evaluate(np.full(383, 4))
        with self.assertRaises(ValueError): native.evaluate(np.zeros(383))


if __name__ == '__main__': unittest.main()
