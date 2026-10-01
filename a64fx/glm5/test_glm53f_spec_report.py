"""Reject incomplete or mislabeled speculative measurements, using repo scratch."""
import json
from pathlib import Path
import tempfile
import unittest
from report_glm53f_spec_runs import report

REPO = Path(__file__).resolve().parents[2]


class SpecReportTest(unittest.TestCase):
    def setUp(self):
        (REPO / 'tmp').mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=REPO / 'tmp')
        self.addCleanup(self.temp.cleanup)
        self.prefix = Path(self.temp.name) / 'run.ids'
        self.log = Path(self.temp.name) / 'run.log'
        self.reference = list(range(100, 120))
        Path(str(self.prefix) + '.greedy').write_text('\n'.join(map(str, self.reference)))
        self.rows = []
        self.complete = dict(variants=2, repetitions=3, cycles=4, status='PASS')
        for variant, depth in enumerate((1, 2)):
            for trial in range(-1, 3):
                accepted = trial + 2
                delivered = 8 + accepted
                self.rows.append(dict(variant=variant, trial=trial, drafts=depth, warmup=trial == -1,
                    accepted=accepted, proposed=4 * depth, delivered=delivered,
                    reference_checked=delivered, minimum_available_kb=3 * 1024 * 1024,
                    seconds=0.1, tok_s=delivered / 0.1, status='PASS'))
                Path(f'{self.prefix}.v{variant}.d{depth}.trial{trial}').write_text('\n'.join(map(str, self.reference[:delivered])))

    def run_report(self):
        lines = ['GLM53F_SPEC_TRIAL ' + json.dumps(t) for t in self.rows]
        if self.complete is not None:
            lines.append('GLM53F_SPEC_COMPLETE ' + json.dumps(self.complete))
        self.log.write_text('\n'.join(lines))
        return report(self.log, self.prefix, 4, [1, 2])

    def test_complete_exact_sweep(self):
        result = self.run_report()
        self.assertTrue(result['greedy_exact'])
        self.assertEqual([v['median_tok_s'] for v in result['variants']], [110, 110])

    def test_partial_or_unverified_trials(self):
        for field, value in [('reference_checked', 0), ('minimum_available_kb', 1),
                             ('status', 'FAIL'), ('tok_s', float('nan')), ('seconds', 2),
                             ('delivered', 20), ('proposed', 3), ('warmup', True)]:
            with self.subTest(field=field):
                old = self.rows[-1][field]
                self.rows[-1][field] = value
                with self.assertRaises(ValueError): self.run_report()
                self.rows[-1][field] = old

    def test_missing_or_duplicate_trials(self):
        last = self.rows.pop()
        with self.assertRaises(ValueError): self.run_report()
        self.rows.append(last)
        self.rows.append(dict(last))
        with self.assertRaises(ValueError): self.run_report()

    def test_token_mismatch(self):
        Path(f'{self.prefix}.v1.d2.trial2').write_text('999\n')
        with self.assertRaises(ValueError): self.run_report()

    def test_missing_depth(self):
        self.rows = self.rows[:4]
        with self.assertRaises(ValueError): self.run_report()

    def test_missing_completion(self):
        self.complete = None
        with self.assertRaises(ValueError): self.run_report()


if __name__ == '__main__':
    unittest.main()
