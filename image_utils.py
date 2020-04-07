import numpy as np

def xyz_to_lms(im: np.ndarray):
    """
    Converts a given images from XYZ format to LMS format, using Hunt-Pointer-Estevez


    :param im:
    :return:
    """
    # raise Exception("not implemented")

    M = np.array([
        0.3897, 0.6889, -0.0786,
        -0.2298, 1.1834, 0.0464,
        0.0000, 0.0000, 1.000,
    ]).reshape((3, 3))


