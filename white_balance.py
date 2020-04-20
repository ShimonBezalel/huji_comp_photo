import numpy
from image_utils import *
import matplotlib.pyplot as plt

from contants import *

from medpy.filter.smoothing import anisotropic_diffusion as ad


def anisotropic_diffusion(im: np.ndarray, gamma: float=0.2):
    """
    Anisotrpic Diffusion is a technique aiming at reducing image noise without removing significant
    parts of the image content. https://en.wikipedia.org/wiki/Anisotropic_diffusion


    :param im: an image to blur, can include all channels
    :param gamma: a value between 0.01 and 0.25, representing how quickly the objects diffuse in the image
    :return: an image in the same shape
    """
    return ad(im)


def correct_white_balance(im_no_flash, im_flash, flash_chromaticity=DEFAULT_CHROMATICITY):
    """
    Given two images of identical scenes, perform a white-balance improvement on the no-flash image,
    using the flash's chromaticity. The two images must be taken using manual settings, so that only
    the flash acts as a controlled difference between them.

    Proper white balance is achieved by inverting the light source color.

    :param im_flash:
    :param im_no_flash:
    :return:
    """
    # 1. Subtract no-flash image from flash image. Result holds D_p = R_p*k_p*Fc
    diff: np.ndarray = im_flash - im_no_flash

    # 2. Calculate the general intensities of brightness throughout the image (k_p).
    norm_method = 2# Current method relying on norms
    intensities = np.linalg.norm(diff, ord=norm_method, axis=2, keepdims=True)



    raise NotImplemented

def run():
    image = imageio.imread('C:\other\huji_comp_photo\input\input-tiff\graycard.tiff')
    calculate_chromaticity(image)


if __name__ == '__main__':
    run()