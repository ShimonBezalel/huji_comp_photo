import numpy as np
import matplotlib.pyplot as plt
from os import path

from contants import *
from image_utils import *

from skimage import img_as_float
from skimage.filters import gaussian
from scipy.ndimage import gaussian_filter
from skimage.feature import canny

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
    diff: np.ndarray = (im_flash - im_no_flash)

    # 2 Remove flash chromaticity by dividing it out, DNC_p = D_p/C_f = R_p*k_p
    vec = chromaticity_to_vector(flash_chromaticity)
    diff_no_color = diff / vec

    flash_intensities = image_intensities(diff_no_color)

    # 3. Estimate reflective spectrum of objects, by dividing: R_p = DNC_p / k_p
    R_colors = diff_no_color / (flash_intensities + 0.0000001)

    # 3.1 The reflective colors can be smoothed as well, considering natural colors of objects are continuous
    R_smoothed = anisotropic_diffusion(R_colors, option=2)

    # 3.2 Balance reflective color range, by simple stretching to 0 -> 1
    R = stretch(R_smoothed)

    # 3.3 Estimate hole regions in R by creating a mask from flash burns and shadows
    light_source_intensities = image_intensities(im_no_flash)

    sigma = 7
    color_edges = np.zeros((im_no_flash.shape[:2])).astype(np.bool)
    for channel in range(3):
        c = R_colors
        color_edges |= canny(np.squeeze(c[..., channel]), sigma=sigma)
    if DEBUG:
        plt.imshow(np.squeeze(color_edges), cmap='gray')
        plt.show()

    error_intensities: np.ndarray = stretch(flash_intensities - light_source_intensities)

    R_ = R

    if DEBUG:
        plt.imshow(np.squeeze(error_intensities))
        plt.show()
    flash_specular_mask = generate_percentage_mask(error_intensities, percentage=0.02, smoothing_sigma=2)
    shadow_area_mask = generate_percentage_mask((1-error_intensities), percentage=0.3, smoothing_sigma=2)
    for mask in [flash_specular_mask, shadow_area_mask]:
        mask = np.squeeze(mask) & (~color_edges)
        plt.imshow(np.squeeze(mask))
        plt.show()
        R_ = fill_holes(R, mask, spatial_color_map=im_no_flash)
        plt.imshow(np.squeeze(np.abs(R - R_)))
        plt.show()
        plt.imshow(np.squeeze(stretch(R - R_) ** 2))
        plt.show()

    # if DEBUG:
    #     plt.imshow(np.squeeze(mask))
    #     plt.show()
    #
    # new_R = fill_holes(R, mask, color_comparer=im_no_flash)

    if DEBUG:
        plt.imshow(np.squeeze(R_))
        plt.show()



    # 4. Assuming no-flash image's illuminant has its own chromaticity and intensities, but the same
    # reflective qualities (R), we can now divide out the reflectiveness I_p = R_p * k2_p * C_nf

    with_R = light_source_intensities * (R ** 4)
    plt.imshow(stretch(with_R))
    plt.show()
    wb_im = light_source_intensities * (R_ ** 4)
    plt.imshow(stretch(wb_im))
    plt.show()
    wb_im = light_source_intensities * (R_)
    plt.imshow(stretch(wb_im ))
    plt.show()
    plt.imshow(im_no_flash)
    plt.show()


    wb_im = (light_source_intensities ** 0.8) * (R_ ** 1.5)
    plt.imshow(wb_im)
    plt.show()

    # wb_im = light_source_intensities * (new_R ** 2)
    # plt.imshow(stretch(wb_im ** 0.3))
    # plt.show()





def run():
    # image = imageio.imread('C:\other\huji_comp_photo\input\input-tiff\graycard.tiff')

    im_name = "im2_"
    base_path = path.join('input', 'input-tiff')
    im_ext = ".JPG"

    path_noflash_image = path.join(base_path, "{}noflash{}".format(im_name, im_ext))
    path_withflash_image = path.join(base_path, "{}withflash{}".format(im_name, im_ext))
    path_graycard_image = path.join(base_path, "{}graycard{}".format(im_name, im_ext))

    im_graycard = img_as_float(imageio.imread(path_graycard_image))
    im_noflash = img_as_float(imageio.imread(path_noflash_image))
    im_withflash = img_as_float(imageio.imread(path_withflash_image))

    chromaticity = calculate_chromaticity(stretch(im_graycard))

    im_noflash = im_noflash[0:3200, ...]
    im_withflash = im_withflash[0:3200, ...]

    res = correct_white_balance(stretch(im_noflash), stretch(im_withflash), flash_chromaticity=chromaticity)
    # plt.imshow(im_noflash)
    # plt.show()
    # h, e = np.histogramdd(im_noflash.reshape(-1,3), bins=8)
    # histogram3dplot(h, e)
    # plt.show()

if __name__ == '__main__':
    run()
