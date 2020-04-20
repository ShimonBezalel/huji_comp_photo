from unittest import TestCase
import numpy as np
import matplotlib.pyplot as plt

from image_utils import calculate_chromaticity

SHOW_PLOTS = False

class TestCalculate_chromaticity(TestCase):
    def test_synthetic_image(self):
        im_size = (100, 100, 3)

        for gray_scale in np.linspace(0.3, 0.7, num=50):
            synthetic_gray_image : np.ndarray = gray_scale * np.ones(im_size)

            noisiness = 0.02
            noise = np.random.uniform(low=-noisiness, high=noisiness, size=im_size)

            chromaticity_range = 0.06

            flash_chromaticity = np.random.uniform(low=1 - chromaticity_range,
                                                   high=1 + chromaticity_range,
                                                   size=(3, ))

            flashed_image = synthetic_gray_image * flash_chromaticity

            flash_chromaticity /= flash_chromaticity[2]

            noisy_image = flashed_image + noise

            if SHOW_PLOTS:
                plt.imshow(noisy_image)
                plt.show()

            result = calculate_chromaticity(noisy_image)
            normalize = result / result[2]
            print("Grayscale: {}".format(round(gray_scale, 2)))
            print("Flash Chromaticity: {}".format(flash_chromaticity))
            print("Result Chromaticity: {}".format(normalize))
            print("--------------")

            np.testing.assert_almost_equal(flash_chromaticity, normalize, decimal=1)

    def test_real_image(self):
        raise NotImplemented
