from unittest import TestCase
import numpy as np
from image_utils import rgb_to_lms, lms_to_rgb

class TestRGB_TO_LMS(TestCase):
    def test_conversion(self):
        for grayscale in np.linspace(0, 1, 11):
            gray = grayscale * np.ones((1, 1, 3))

            lms_gray = rgb_to_lms(gray)

            back_gray = lms_to_rgb(lms_gray)

            self.assertAlmostEqual(0, np.max(np.abs(gray - back_gray)), delta=0.02)

        color = np.random.uniform(0, 1, 3).reshape((1,1,3))
        as_lms = rgb_to_lms(color)
        back = lms_to_rgb(as_lms)
        self.assertAlmostEqual(0, np.max(np.abs(color - back)), delta=0.02)
