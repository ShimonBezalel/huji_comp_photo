import numpy as np
import matplotlib.pyplot as plt
from os import path

from constants import *
from image_utils import *

from skimage import img_as_float
from skimage.filters import gaussian
from scipy.ndimage import gaussian_filter
from skimage.feature import canny

from medpy.filter.smoothing import anisotropic_diffusion
import cv2

DEBUG = True


def correct_white_balance(im_no_flash: np.ndarray, im_flash: np.ndarray,
                          flash_chromaticity: np.ndarray = GRAY_CHROMATICITY,
                          flash_regions=0.01, shadow_regions=0.15,
                          brightness=0.6, saturation=0.7):
    """
    Given two images of identical scenes, perform a white-balance improvement on the no-flash image,
    using the flash's chromaticity. The two images must be taken using manual settings, so that only
    the flash acts as a controlled difference between them.

    Proper white balance is achieved by inverting the light source color.
    This algorithm corrects regions of error caused by the flash, and provided parameters to control results.
    The result can be brightened or saturated as well.

    :param im_no_flash: The image to be corrected
    :param im_flash: The reference image with flash turned on. No other difference should exists between images,
                    especially no movement of the camera.
    :param flash_chromaticity: Vector of length 2. The estimated color of the flash. Use the provided utility to measure chromaticity.
    :type flash_chromaticity: np.ndarray of shape (2,) or (2,1)

    The following parameters are given
    :param shadow_regions:
    :param flash_regions:

    Control over brightness and saturation.
    Takes values from 0 (dark/pale) to 1 (light/vivid), with input of 0.5 for no affect.
    Note that a balance between these two parameters can lead to better results and obscure errors.
    :param saturation: How strong should the colors come through after white balance correction?
    :param brightness: How bright should the result be?

    :return:
    """
    assert np.all([0 < percentage < 1 for percentage in [flash_regions, shadow_regions]]), ERROR_MSG_PERCENTAGE
    assert np.all([0 < param < 1 for param in [brightness, saturation]]), ERROR_MSG_PARAMETERS

    # 1. Subtract no-flash image from flash image. Result holds D_p = R_p*k_p*C_f
    diff: np.ndarray = (im_flash - im_no_flash)

    # 2 Remove flash chromaticity by dividing it out, DNC_p = D_p/C_f = R_p*k_p
    vec = chromaticity_to_vector(flash_chromaticity)
    diff_no_color = diff / vec

    # 3. Estimate reflective spectrum of objects
    # 3.1 Calculate intensities of both the flash and second light source
    flash_intensities = image_intensities(diff_no_color)
    light_source_intensities = image_intensities(im_no_flash)

    # 3.2 Estimate R by dividing: R_p = DNC_p / k_p
    R_noisy = diff_no_color / (flash_intensities + EPSILON)

    # 3.3 Smooth the reflective colors, assuming natural colors of objects are continuous
    R_smoothed = anisotropic_diffusion(R_noisy, option=2)
    R = normalize(R_smoothed)

    # 3.4 Estimate hole regions in R by creating a mask from flash burns and shadows, and then correct chromatically
    # 3.4.1 Errors are light and shadowy regions that are the difference of the two images
    error_intensities: np.ndarray = normalize(flash_intensities - light_source_intensities)
    flash_specular_mask = generate_percentage_mask(error_intensities, percentage=0.01, smoothing_sigma=1)
    shadow_area_mask = generate_percentage_mask((1 - error_intensities), percentage=0.15, smoothing_sigma=1)

    # 3.4.2 Canny object edge detection. The mask must respect the edges of the objects in
    # order to correct for each separately.
    color_edges = np.zeros((im_no_flash.shape[:2])).astype(np.bool)
    for channel in range(3):
        color_edges |= canny(np.squeeze(R_noisy[..., channel]), sigma=7)  # best with noisy R with very high SD of 7

    # 3.4.3 Fill in error regions with best match colors
    for mask in [flash_specular_mask, shadow_area_mask]:
        mask = np.squeeze(mask) & (~color_edges)  # Remove edges to separate sub-regions
        R = fill_holes(R, mask, spatial_color_map=im_no_flash)

    # 4. Assuming R is same in both images we can now recalculate the WB image I_p = R_p * k2_p
    # 4.1 Some artificial balancing between these two leads to better results.
    brightness_exp = linear(brightness, BRIGHTNESS_MIN, BRIGHTNESS_MAX)
    saturation_exp = linear(2 * saturation - 1, 1, SATURATION_MAX) if saturation > 0.5 \
        else linear(2 * saturation, SATURATION_MIN, 1)
    wb_im = (light_source_intensities ** brightness_exp) * (R ** saturation_exp)
    return wb_im


def run():
    im_name = ""
    base_path = path.join('input', 'input-tiff')
    im_ext = ".tiff"

    path_noflash_image = path.join(base_path, "{}noflash{}".format(im_name, im_ext))
    path_withflash_image = path.join(base_path, "{}withflash{}".format(im_name, im_ext))
    path_graycard_image = path.join(base_path, "{}graycard{}".format(im_name, im_ext))

    im_graycard = img_as_float(read_image_as_lms(path_graycard_image))
    im_noflash = img_as_float(read_image_as_lms(path_noflash_image))
    im_withflash = img_as_float(read_image_as_lms(path_withflash_image))

    chromaticity = calculate_chromaticity(normalize(im_graycard))

    res = correct_white_balance(normalize(im_noflash), normalize(im_withflash), flash_chromaticity=chromaticity)
    res_rgb = lms_to_rgb(res)
    plt.imshow(res_rgb)
    plt.show()


if __name__ == '__main__':
    run()
