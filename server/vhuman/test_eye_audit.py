import unittest

from server.vhuman.metahuman.eye_audit import compare_audits


def row(sid, side, ue, cycles, good=True):
    return dict(sid=sid, side=side, good=good, hsv=dict(ue=ue, cycles=cycles))


class EyeAuditComparisonTest(unittest.TestCase):
    def test_matches_eyes_and_uses_circular_hue(self):
        baseline = dict(rows=[row('a', 'left', [359, .8, .4], [1, .4, .4]),
                              row('b', 'left', [10, .8, .4], [10, .8, .4])])
        candidate = dict(rows=[row('a', 'left', [359, .8, .4], [355, .7, .4]),
                               row('b', 'left', [10, .8, .4], [10, .8, .4], good=False)])
        result = compare_audits(baseline, candidate)
        self.assertEqual(result['summary']['matched_eyes'], 1)
        self.assertEqual(result['summary']['candidate_saturation_closer'], 1)
        self.assertEqual(result['rows'][0]['baseline']['hue_error_degrees'], 2)
        self.assertEqual(result['rows'][0]['candidate']['hue_error_degrees'], 4)

    def test_grey_hue_and_empty_pairs(self):
        a = dict(rows=[row('a', 'right', [90, .05, .4], [180, .8, .4])])
        result = compare_audits(a, a)
        self.assertIsNone(result['rows'][0]['baseline']['hue_error_degrees'])
        self.assertEqual(result['summary']['baseline']['hue_eyes'], 0)
        self.assertIsNone(compare_audits(a, dict(rows=[]))['summary']['baseline']['median_saturation_error'])

    def test_each_variant_uses_its_own_reference_sample(self):
        a = dict(rows=[row('a', 'left', [30, .8, .4], [30, .4, .4])])
        b = dict(rows=[row('a', 'left', [30, .5, .4], [30, .5, .4])])
        result = compare_audits(a, b)
        self.assertAlmostEqual(result['rows'][0]['baseline']['saturation_error'], .4)
        self.assertEqual(result['rows'][0]['candidate']['saturation_error'], 0)


if __name__ == '__main__':
    unittest.main()
