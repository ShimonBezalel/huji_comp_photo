import numpy as np
import matplotlib.pyplot as plt
from os import path

from contants import *
from image_utils import *

from skimage import img_as_float
from skimage.filters import gaussian
from scipy.ndimage import gaussian_filter
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
    # Squaring the difference improves results
    diff: np.ndarray = (im_flash - im_no_flash) ** 2

    # 1.1 Remove flash chromaticity by dividing it out, DNC_p = D_p/C_f = R_p*k_p
    vec = chromaticity_to_vector(flash_chromaticity)
    diff_no_color = diff / vec

    # 2. Calculate the general intensities of brightness throughout the image (k_p).
    # 2.1 Reduce dimension to attempt to find intensity only
    norm_method = np.inf  # Current method relying on norms. Options are [0, 1, 2, np.inf]
    intensities = np.linalg.norm(diff_no_color, ord=norm_method, axis=2, keepdims=True)
    intensities = stretch(intensities)

    # 2.2 Assume light intensity is continuous but retain features.
    # Done using a anisotropic diffusion - a technique aiming at reducing image noise without removing significant
    # parts of the image content. https://en.wikipedia.org/wiki/Anisotropic_diffusion
    intensities_smoothed = anisotropic_diffusion(intensities)

    # 3. Estimate reflective colors of objects, by dividing: R_p = DNC_p / k_p
    R_colors = diff_no_color / intensities_smoothed

    # 3.1 The reflective colors can be smoothed as well, considering natural colors of objects are continuous
    R_smoothed = anisotropic_diffusion(R_colors, option=2)

    # 3.2 Balance reflective color range, by simple stretching to 0 -> 1
    R = stretch(R_smoothed)
    if DEBUG:
        plt.imshow(R)
        plt.show()

    # 3.3 Estimate hole regions in R by creating a mask from specular flash reflections
    flash_percentage = 0.003  # % of pixels are flash specular reflection
    hist, bins = np.histogram(intensities, bins=255)
    dark_normalized_cumsum = np.cumsum(hist) / intensities.size
    light_normalized_cumsum = np.cumsum(np.flip(hist)) / intensities.size
    flash_intensity_threshold = np.flip(bins)[np.argmax(np.where(light_normalized_cumsum < flash_percentage))]
    holes = np.array(intensities >= flash_intensity_threshold).astype(np.float64)
    holes_smooth = gaussian_filter(holes, sigma=2)
    holes_mask = holes_smooth > 0.001
    # res = R
    res = stretch(fill_holes(R, mask=np.squeeze(holes_mask)))
    # if DEBUG:
    #     plt.imshow(stretch(res))
    #     plt.show()
    #     plt.imshow(R)
    #     plt.show()
    # shade_region = (0.02, 0.1)
    # shade_percentage = 0.1
    # shade_threshold = bins[np.argmax(np.where(dark_normalized_cumsum < shade_percentage))]
    most_common_color = get_most_common_color(R)
    wall_color = most_common_color
    if DEBUG:
        plt.imshow(wall_color)
        plt.show()
    low_pass = gaussian_filter(intensities, sigma=4)
    if DEBUG:
        plt.imshow(np.squeeze(low_pass))
        plt.show()
    shades = ((low_pass >= 0.00015) & (low_pass <= 0.00025))
    # shades = binary_closing(shades, iterations=1)
    shades = shades.astype(np.float64)

    shades_smooth = gaussian_filter(shades, sigma=0.5)
    shades_mask = shades_smooth > 0.001
    if DEBUG:
        plt.imshow(np.squeeze(shades_mask))
        plt.show()
    res = fill_holes(res, mask=np.squeeze(shades_mask), hole_percentage_threshold=0.005)
    if DEBUG:
        plt.imshow(stretch(res))
        plt.show()
        plt.imshow(R)
        plt.show()

    return

    # 4. Assuming no-flash image's illuminant has its own chromaticity and intensities, but the same
    # reflective qualities (R), we can now divide out the reflectiveness I_p = R_p * k2_p * C_nf
    raise NotImplemented





def run():
    # image = imageio.imread('C:\other\huji_comp_photo\input\input-tiff\graycard.tiff')

    im_name = ""
    base_path = path.join('input', 'input-tiff')

    path_noflash_image = path.join(base_path, "{}noflash.tiff".format(im_name))
    path_withflash_image = path.join(base_path, "{}withflash.tiff".format(im_name))
    path_graycard_image = path.join(base_path, "{}graycard.tiff".format(im_name))

    im_graycard = img_as_float(imageio.imread(path_graycard_image))
    im_noflash = img_as_float(imageio.imread(path_noflash_image))
    im_withflash = img_as_float(imageio.imread(path_withflash_image))

    chromaticity = calculate_chromaticity(im_graycard)

    res = correct_white_balance(im_noflash, im_withflash, flash_chromaticity=chromaticity)
    # plt.imshow(im_noflash)
    # plt.show()
    # h, e = np.histogramdd(im_noflash.reshape(-1,3), bins=8)
    # histogram3dplot(h, e)
    # plt.show()

if __name__ == '__main__':
    run()
