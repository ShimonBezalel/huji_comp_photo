import numpy as np
import matplotlib.pyplot as plt
from os import path

from contants import *
from image_utils import *

from skimage import img_as_float
from skimage.filters import gaussian
from medpy.filter.smoothing import anisotropic_diffusion


DEBUG = True


def correct_white_balance(im_no_flash, im_flash, flash_chromaticity=NO_CHROMATICITY):
    """
    Given two images of identical scenes, perform a white-balance improvement on the no-flash image,
    using the flash's chromaticity. The two images must be taken using manual settings, so that only
    the flash acts as a controlled difference between them.

    Proper white balance is achieved by inverting the light source color.

    :param im_flash:
    :param im_no_flash:
    :return:
    """
    # 1. Subtract no-flash image from flash image. Result holds D_p = R_p*k_p*C_f
    diff: np.ndarray = im_flash - im_no_flash

    # 1.1 Remove flash chromaticity by dividing it out, DNC_p = D_p/C_f = R_p*k_p
    vec = chromaticity_to_vector(flash_chromaticity)
    diff_no_color = diff / vec

    # 2. Calculate the general intensities of brightness throughout the image (k_p).
    # 2.1 Reduce dimension to attempt to find intensity only
    norm_method = 2  # Current method relying on norms. Options are [0, 1, 2, np.inf]
    intensities = np.linalg.norm(diff_no_color, ord=norm_method, axis=2, keepdims=True)

    # 2.2 Assume light intensity is continuous but retain features.
    # Done using a anisotropic diffusion - a technique aiming at reducing image noise without removing significant
    # parts of the image content. https://en.wikipedia.org/wiki/Anisotropic_diffusion
    intensities_smoothed = anisotropic_diffusion(intensities)

    # 3. Estimate reflective colors of objects, by dividing - R_p = DNC_p / k_p
    R_colors = diff_no_color / intensities_smoothed

    # 3.1 The reflective colors can be smoothed as well, considering natural colors of objects are continuous
    R_smoothed = anisotropic_diffusion(R_colors, option=2)
    R_s2 = gaussian(R_colors, sigma=2)

    if DEBUG:
        plt.imshow(R_smoothed)
        plt.show()
        plt.imshow(R_s2)
        plt.show()

    # 3.2 Balance reflective color range, by simple stretching to 0 -> 1
    R = (R_smoothed - np.min(R_smoothed)) / np.ptp(R_smoothed)

    if DEBUG:
        plt.imshow(R)
        plt.show()

    # 4. Assuming no-flash image's illuminant has its own chromaticity and intensities, but the same
    # reflective qualities, we can now divide out the reflectiveness I_p = R_p * k2_p * C_nf
    raise NotImplemented





def run():
    # image = imageio.imread('C:\other\huji_comp_photo\input\input-tiff\graycard.tiff')

    im_name = ""
    base_path = path.join('input', 'input-tiff')

    path_noflash_image = path.join(base_path, "{}noflash.tiff".format(im_name))
    path_withflash_image = path.join(base_path, "{}withflash.tiff".format(im_name))
    path_graycard_image = path.join(base_path, "{}graycard.tiff".format(im_name))

    im_graycard = stretch(img_as_float(imageio.imread(path_graycard_image)))
    im_noflash = stretch(img_as_float(imageio.imread(path_noflash_image)))
    im_withflash = stretch(img_as_float(imageio.imread(path_withflash_image)))

    chromaticity = calculate_chromaticity(im_graycard)

    res = correct_white_balance(stretch(im_noflash), stretch(im_withflash), flash_chromaticity=chromaticity)





if __name__ == '__main__':
    run()
