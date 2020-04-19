from unittest import TestCase
import numpy as np
import matplotlib.pyplot as plt

from image_utils import calculate_chromaticity

class TestCalculate_chromaticity(TestCase):
    def test_synthetic_image(self):
        im_size = (100, 100, 3)

        w = 10
        h = 10
        fig = plt.figure(figsize=(8, 8))
        columns = 3
        rows = 10
        for gray_scale in np.linspace(0.1, 0.9, num=rows):
            synthetic_gray_image : np.ndarray = gray_scale * np.ones(im_size)

            noise = np.random.uniform(low=-0.05, high=0.05, size=im_size)

            flash_chromaticity = np.random.uniform(low=-0.1, high=0.1, size=(1, 3))

            flashed_image = synthetic_gray_image + flash_chromaticity

            noisy_image = flash_chromaticity + noise

            result = calculate_chromaticity(noisy_image)

            self.assertAlmostEqual(flash_chromaticity, result, places=1)
