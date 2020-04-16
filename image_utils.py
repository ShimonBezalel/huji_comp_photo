import numpy as np
import rawpy
import imageio

def xyz_to_lms(im: np.ndarray):
    """
    Converts a given images from XYZ format to LMS format, using Hunt-Pointer-Estevez

    :param im:
    :return:
    """

    M = np.array([
        0.3897, 0.6889, -0.0786,
        -0.2298, 1.1834, 0.0464,
        0.0000, 0.0000, 1.000,
    ]).reshape((3, 3))

    raise Exception("not implemented")



def calculate_chromaticity(im):
    """
    Given an image of only a gray-card, calculate the chromaticity of the camera's flash + setting
    :param im: Image should be simple RAW format image of just a gray-card. If more than a gray-card was
    in the frame of the original image, crop the rest out.
    :return: a value between 0 and 1 representing the chromaticity
    """
    raise NotImplemented


def save_linear_image(path):
    with rawpy.imread(path) as raw:
        rgb = raw.postprocess(gamma=(1, 1), no_auto_bright=True, output_bps=16)
    imageio.imsave('samples/linear.tiff', rgb)

def open_raw(path):
    """
    Opens a raw image
    """
    with rawpy.imread(path) as raw:
        # rgb = raw.postprocess()
        raw_image = raw.raw_image.copy()
        return raw_image
