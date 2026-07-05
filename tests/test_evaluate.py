"""Unit tests for the evaluation logic.

The evaluation module is the one place where a silent bug would corrupt
every reported number (F1, HitRate), so its pure functions get direct
tests. Run with:

    python -m unittest discover tests
"""

import unittest

import numpy as np

from crashlens.detect import merge_segments
from crashlens.evaluate import hitrate, label_segments, point_adjust, prf


class TestLabelSegments(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(label_segments(np.array([0, 0, 0])), [])

    def test_single_segment(self):
        self.assertEqual(label_segments(np.array([0, 1, 1, 0])), [(1, 2)])

    def test_multiple_segments(self):
        labels = np.array([1, 0, 1, 1, 0, 1])
        self.assertEqual(label_segments(labels), [(0, 0), (2, 3), (5, 5)])

    def test_segment_reaching_end(self):
        # a run of 1s ending at the last index must still be closed
        self.assertEqual(label_segments(np.array([0, 1, 1])), [(1, 2)])


class TestPointAdjust(unittest.TestCase):
    def test_one_hit_credits_whole_segment(self):
        labels = np.array([0, 1, 1, 1, 0])
        pred = np.array([0, 0, 1, 0, 0])  # only one point detected
        adjusted = point_adjust(pred, labels)
        np.testing.assert_array_equal(adjusted, [0, 1, 1, 1, 0])

    def test_missed_segment_stays_missed(self):
        labels = np.array([0, 1, 1, 0])
        pred = np.array([0, 0, 0, 0])
        np.testing.assert_array_equal(point_adjust(pred, labels), pred)

    def test_false_positives_untouched(self):
        # adjustment must never remove predictions outside true segments
        labels = np.array([0, 0, 1, 1])
        pred = np.array([1, 0, 1, 0])
        np.testing.assert_array_equal(point_adjust(pred, labels), [1, 0, 1, 1])


class TestPRF(unittest.TestCase):
    def test_perfect(self):
        labels = np.array([0, 1, 1, 0])
        p, r, f1 = prf(labels, labels)
        self.assertEqual((p, r, f1), (1.0, 1.0, 1.0))

    def test_no_predictions_is_zero_not_crash(self):
        p, r, f1 = prf(np.array([0, 0]), np.array([1, 1]))
        self.assertEqual((p, r, f1), (0.0, 0.0, 0.0))


class TestHitRate(unittest.TestCase):
    def test_full_recovery(self):
        # top-|GT| of the ranking contains all ground-truth dims
        self.assertEqual(hitrate([3, 1, 2, 0], [1, 3], 100), 1.0)

    def test_partial_recovery(self):
        self.assertEqual(hitrate([3, 0, 1, 2], [1, 3], 100), 0.5)

    def test_150_percent_widens_the_window(self):
        ranking, gt = [5, 0, 1, 2], [1, 5]
        self.assertEqual(hitrate(ranking, gt, 100), 0.5)  # top-2: [5, 0]
        self.assertEqual(hitrate(ranking, gt, 150), 1.0)  # top-3: [5, 0, 1]


class TestMergeSegments(unittest.TestCase):
    def test_close_segments_merge(self):
        self.assertEqual(merge_segments([(0, 2), (4, 6)], gap=2), [(0, 6)])

    def test_distant_segments_stay_apart(self):
        self.assertEqual(merge_segments([(0, 2), (10, 12)], gap=2),
                         [(0, 2), (10, 12)])


if __name__ == "__main__":
    unittest.main()
