import unittest

import numpy as np

from analyze import estimate


class PairedBootstrapTest(unittest.TestCase):
    def test_identical_conditions_have_exactly_zero_paired_difference(self):
        videos = np.array([0.0, 0.25, 0.75, 1.0])
        point, low, high = estimate(videos - videos)
        self.assertEqual((point, low, high), (0.0, 0.0, 0.0))

    def test_constant_video_difference_is_preserved(self):
        point, low, high = estimate(np.full(73, 0.125))
        self.assertEqual((point, low, high), (12.5, 12.5, 12.5))


if __name__ == '__main__':
    unittest.main()
