import numpy as np
import matplotlib.pyplot as plt
from os import path
import argparse

from constants import *
from image_utils import *

from skimage import img_as_float
from skimage.feature import canny

from medpy.filter.smoothing import anisotropic_diffusion

DEBUG = True



def correct_white_balance(im_no_flash: np.ndarray, im_flash: np.ndarray,
                          flash_chromaticity: np.ndarray = GRAY_CHROMATICITY,
                          flash_regions: float = PERCENT_FLASH_DEFAULT, shadow_regions: float = PERCENT_SHADOW_DEFAULT,
                          brightness: float = BRIGHTNESS_DEFAULT, saturation: float = SATURATION_DEFAULT):
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
    flash_specular_mask = generate_percentage_mask(error_intensities, percentage=flash_regions, smoothing_sigma=1)
    shadow_area_mask = generate_percentage_mask((1 - error_intensities), percentage=shadow_regions, smoothing_sigma=1)

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


def run(**kwargs):
    im_noflash, im_withflash, im_graycard = (
        img_as_float(imageio.imread(kwargs["image_path_{}".format(suffix.value)]))
        for suffix in IMAGE_SUFFIX
    )

    chromaticity = calculate_chromaticity(im_graycard)

    res = correct_white_balance(im_noflash, im_withflash, flash_chromaticity=chromaticity,
                                saturation=kwargs["saturation"], brightness=kwargs["brightness"],
                                shadow_regions=kwargs["shadow_regions"], flash_regions=kwargs["flash_regions"])
    if (kwargs["out_path"]):
        with open(kwargs["out_path"], 'w') as f:
            plt.imsave(f, res)
        out_dir, out_basename = path.split(kwargs["out_path"])

        with open(path.join(out_dir, "gamma_corrected_" + out_basename), 'w') as f:
            plt.imsave(f, res ** 0.3)
    else:
        plt.imshow(res)
        plt.show()
        plt.imshow(res ** 0.3)
        plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="""White Balance from two images, flash and no flash. 
    Given two images of identical scenes, perform a white-balance improvement on the no-flash image,
    using the flash's chromaticity. The two images must be taken using manual settings, so that only
    the flash acts as a controlled difference between them.""")

    parser.add_argument("-p", "--path", default="input/input-tiff/", dest="full_path", help="""
    Path must be either the name of dir or path to one of the images.
    If image file path is provided, provide one image path such as im1_{suffix}.JPG. The other two images will follow 
    the same format - <name><suffix>.<extension>.
    If path to directory is provided, supply a directory with three images containing the default names:\n{all_suffixes}
    """.format(suffix=IMAGE_SUFFIX.NO_FLASH.value,
               all_suffixes=", ".join(["{}{}.{}".format(DEFAULT_NAME, suffix.value, DEFAULT_EXT)
                                      for suffix in IMAGE_SUFFIX])))

    parser.add_argument("-fp", "--flash", default=PERCENT_FLASH_DEFAULT, dest="flash_regions", type=float, help="""
    Optional control over estimated spectral flash reflections' percentage of pixels. Defaults to {}, or {}%%.
     Takes values from 0 (No reflections caused by the cameras flash, such as a open-space scene 
     or non-reflective materials) to 1 (The entire image is a flash reflection. Not very likely.)
     Flash regions are normally small, therefore an input of less than 1%%, or 0.01 is reasonable.  
    """.format(PERCENT_FLASH_DEFAULT, PERCENT_FLASH_DEFAULT * 100))

    parser.add_argument("-sp", "--shadow", default=PERCENT_SHADOW_DEFAULT, dest="shadow_regions", type=float, help="""
    Optional control over estimated shadow percentage of pixels. Defaults to {}, or {}%%.
     Takes values from 0 (No shadows caused by the cameras flash, such as a open-space scene with no reflective walls)
     to 1 (The entire image is a shadow caused by the flash. Not very likely.) Shadowy regions are normally large, 
     therefore an input of about 20%%, or 0.2 is reasonable for a portrait in confined space, 
     where the camera's flash is likely to cause shadows.  
    """.format(PERCENT_SHADOW_DEFAULT, PERCENT_SHADOW_DEFAULT * 100))

    parser.add_argument("-b", "--brightness", default=BRIGHTNESS_DEFAULT, dest="brightness", type=float, help="""
    Optional control over brightness. Defaults to {}.
    Takes values from 0 (dark) to 1 (light), with input of 0.5 for no effect.
    Note that a balance between this parameter and saturation  can lead to better results and obscure errors.
    """.format(BRIGHTNESS_DEFAULT))

    parser.add_argument("-s", "--saturation", default=SATURATION_DEFAULT, dest="saturation", type=float, help="""
    Optional control over saturation. Defaults to {}.
    Takes values from 0 (pale) to 1 (vivid), with input of 0.5 for no effect.
    Note that a balance between this parameter and brightness can lead to better results and obscure errors.
    How strong should the colors come through after white balance correction?
    """.format(SATURATION_DEFAULT))

    parser.add_argument("-o", "--out_path", default="", dest="out_path", type=str, help="""
    Optional. Path in which to save result. If not specified, the result will be displayed instead. 
    Two images are saves, one without post-processing <out_path> and one gamma corrected (brighter) 
    <out_dir> + gamma_corrected_ <out_basename>
    """)

    ns = parser.parse_args()

    # Find name of image without suffixes
    basename = path.basename(ns.full_path)
    image_path_provided = basename != ''
    if image_path_provided:
        assert "." in basename, ERROR_MSG_PATH
        im_name, im_ext = path.basename(ns.full_path).split(".")
        for im_type in IMAGE_SUFFIX:
            im_name = im_name.replace(im_type.value, "")
    else:  # dir name provided
        im_name, im_ext = DEFAULT_NAME, DEFAULT_EXT

    image_paths = {
        "image_path_{}".format(suffix): path.join(path.dirname(ns.full_path), "{}{}.{}".format(im_name, suffix, im_ext))
        for suffix in [s.value for s in IMAGE_SUFFIX]}
    for p in image_paths.values():
        assert path.exists(p), "\n".join(["{} does not exist.".format(p), ERROR_MSG_EXISTS, ERROR_MSG_PATH])

    for val in [ns.flash_regions, ns.shadow_regions]:
        assert 0 <= val <= 1, ERROR_MSG_PERCENTAGE
    for val in [ns.saturation, ns.brightness]:
        assert 0 <= val <= 1, ERROR_MSG_PARAMETERS
    ret = dict(ns._get_kwargs())
    ret.update(image_paths)
    return ret


if __name__ == '__main__':
    run(**parse_args())
