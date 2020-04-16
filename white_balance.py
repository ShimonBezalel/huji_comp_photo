import numpy
from image_utils import *
import matplotlib.pyplot as plt

from contants import *


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
    # 1. Subtract no-flash image from flash image
    difference = im_flash - im_no_flash


    raise NotImplemented

def run():
    example = save_linear_image("samples/E1DXINBI000050.CR2")
    plt.imshow(example)
    plt.show()


if __name__ == '__main__':
    run()