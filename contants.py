import numpy as np
import enum


#  Library
# -------------------------------------------------
class IMAGE_SUFFIX(enum.Enum):
    # Warning: Do not change order (white_balance.py - run)
    NO_FLASH    = "noflash"
    WITH_FLASH  = "withflash"
    GRAYCARD    = "graycard"

DEFAULT_EXT     = "tiff"
DEFAULT_NAME    = ""


#  Image processing
# -------------------------------------------------
GRAY_CHROMATICITY   = np.ones((2,))

DELTA_THRESHOLD     = 0.005

BRIGHTNESS_MIN      = 0.5
BRIGHTNESS_MAX      = 1.5
BRIGHTNESS_DEFAULT  = 0.5  # Between 0 and 1


SATURATION_MIN      = 0.4
SATURATION_MAX      = 4
SATURATION_DEFAULT  = 0.6  # Between 0 and 1


PERCENT_FLASH_DEFAULT   = 0.005
PERCENT_SHADOW_DEFAULT  = 0.25


# Math
# -------------------------------------------------
EPSILON = 0.00001

# Error Messages
# -------------------------------------------------
ERROR_MSG_PERCENTAGE    = """
Flash and Shadow region percentages must be values between 0 and 1. 
For example: flash_region=0.01 means 1% of the provided flash image is estimated to be reflective burns."""

ERROR_MSG_PARAMETERS    = """
Optional parameters for saturation and brightness must be values between 0 and 1. 
For example: saturation=0.5 means no changes should be made to results saturation, 
whereas 0 means desaturate fully and 1 means saturate fully."""

ERROR_MSG_PATH          = """
Path must be either the name of dir or path to one of the images.
If image file path is provided, provide one image path such as im1_{suffix}.JPG. The other two images will follow 
the same format. (<name><suffix>.<extension>)
If path to directory is provided, supply a directory with three images containing the default names:\n{all_suffixes}"""\
    .format(suffix=IMAGE_SUFFIX.NO_FLASH.value,
            all_suffixes="\n".join("{}{}.{}".format(DEFAULT_NAME, suffix.value, DEFAULT_EXT) for suffix in IMAGE_SUFFIX))

ERROR_MSG_EXISTS        = """
Path provided does not exist."""

# Algorithms
# -------------------------------------------------
class INTENSITY_METHOD(enum.Enum):
    @staticmethod
    class NORM(enum.Enum):
        L0      = 0
        L1      = 1
        L2      = 2
        L_INF   = np.inf

    # DEFAULT = 2


class CONVERSION_MATRIX(enum.Enum):
    class XYZ_TO_LMS(enum.Enum):

        # Hunt-Pinter-Estevez P.19 https://moodle2.cs.huji.ac.il/nu19/pluginfile.php/482566/mod_resource/content/0
        # /Lecture%202a%20-%20white%20balance.pdf
        HPE = [(
            (0.3897, 0.6889, -0.0786),
            (-0.2298, 1.1834, 0.0464),
            (0.0000, 0.0000, 1.000))]

        # Normalized to D65 https://en.wikipedia.org/wiki/LMS_color_space
        HPE_NORMALIZED = [(
            (0.4002, 0.7075, -0.0807),
            (-0.2280, 1.1500, 0.0612),
            (0.0000, 0.0000, 0.9184))]

        # CIECAM97s = np.array((
        #     (0.4002, 0.7075, -0.0807),
        #     (-0.2280, 1.1500, 0.0612),
        #     (0.0000, 0.0000, 0.9184)))
