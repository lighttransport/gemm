"""Portable source evidence must match the verified geometry and resist tampering."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from .reconstruction.usd_candidate_evidence import attach, digest, verify

WORK = Path(__file__).resolve().parents[2]/'tmp/vhuman-usd-evidence-tests'


class EvidenceTests(unittest.TestCase):
    def fixture(self, root):
        candidate, bundle = root/'candidate', root/'bundle'
        candidate.mkdir(); bundle.mkdir()
        (candidate/'geometry.npz').write_bytes(b'geometry fixture')
        (candidate/'portrait.png').write_bytes(b'portrait fixture')
        manifest = dict(format='vhuman.reconstruction.v1', face_model='gnm_v3', material={},
            geometry_sha256=digest(candidate/'geometry.npz'),portrait_sha256=digest(candidate/'portrait.png'))
        (candidate/'manifest.json').write_text(json.dumps(manifest))
        shutil.copyfile(candidate/'manifest.json',bundle/'candidate_manifest.json')
        (bundle/'head.usdc').write_bytes(b'USD fixture')
        (bundle/'report.json').write_text(json.dumps(dict(passed=True,
            candidate_geometry_sha256=manifest['geometry_sha256'],usd_sha256=digest(bundle/'head.usdc'))))
        return candidate,bundle

    def test_source_subset_survives_relocation_and_rejects_changed_file(self):
        WORK.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=WORK) as directory:
            root=Path(directory);candidate,bundle=self.fixture(root)
            self.assertTrue(attach(candidate,bundle)['passed'])
            relocated=root/'relocated';shutil.move(bundle,relocated)
            self.assertTrue(verify(relocated)['passed'])
            usd=relocated/'head.usdc';original_usd=usd.read_bytes();usd.write_bytes(b'changed USD')
            with self.assertRaisesRegex(ValueError,'USD content hash'):
                verify(relocated)
            usd.write_bytes(original_usd)
            (relocated/'candidate_evidence/portrait.png').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'file mismatch'):
                verify(relocated)

    def test_other_source_and_receipt_tampering_are_rejected(self):
        WORK.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=WORK) as directory:
            candidate,bundle=self.fixture(Path(directory))
            original=(bundle/'candidate_manifest.json').read_bytes()
            (bundle/'candidate_manifest.json').write_bytes(b'another manifest')
            with self.assertRaisesRegex(ValueError,'does not match'):
                attach(candidate,bundle)
            self.assertFalse((bundle/'candidate_evidence').exists())
            (bundle/'candidate_manifest.json').write_bytes(original)
            attach(candidate,bundle)
            (bundle/'candidate_evidence.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'receipt hash'):
                verify(bundle)


if __name__=='__main__':unittest.main()
