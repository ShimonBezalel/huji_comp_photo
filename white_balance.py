import numpy
from image_utils import *


def correct_white_balance(im_no_flash, im_flash):
    """
    Given two images of identical scenes, perform a white-balance improvement on the no-flash image,
    using the flash's chromaticity. The two images must be taken using manual settings, so that only
    the flash acts as a controlled difference between them.

    Proper white balance is achieved by inverting the light source color.

    :param im_flash:
    :param im_no_flash:
    :return:
    """
    raise NotImplemented

