import numpy as np
import enum

#  Image processing
# -------------------------------------------------
GRAY_CHROMATICITY = np.ones((2,))

DELTA_THRESHOLD = 0.005

BRIGHTNESS_MIN = 0.5
BRIGHTNESS_MAX = 1.5

SATURATION_MIN = 0.4
SATURATION_MAX = 4

# Math
# -------------------------------------------------
EPSILON = 0.00001

# Error Messages
# -------------------------------------------------
ERROR_MSG_PERCENTAGE = """
Flash and Shadow region percentages must be values between 0 and 1. 
For example: flash_region=0.01 means 1% of the provided flash image is estimated to be reflective burns.
"""

ERROR_MSG_PARAMETERS = """
Flash and Shadow region percentages must be values between 0 and 1. 
For example: flash_region=0.01 means 1% of the provided flash image is estimated to be reflective burns.
"""


# Algorithms
# -------------------------------------------------
class INTENSITY_METHOD(enum.Enum):
	@staticmethod
	class NORM(enum.Enum):
		L0 = 0
		L1 = 1
		L2 = 2
		L_INF = np.inf

	# DEFAULT = 2


class CONVERSION_MATRIX(enum.Enum):
	HPE = [(
		(0.3897, 0.6889, -0.0786),
		(-0.2298, 1.1834, 0.0464),
		(0.0000, 0.0000, 1.000))]

#
# class XYZ_TO_LMS(enum.Enum):
# 	# Hunt-Pinter-Estevez P.19 https://moodle2.cs.huji.ac.il/nu19/pluginfile.php/482566/mod_resource/content/0
# 	# /Lecture%202a%20-%20white%20balance.pdf
# 	HPE = [(
# 		(0.3897, 0.6889, -0.0786),
# 		(-0.2298, 1.1834, 0.0464),
# 		(0.0000, 0.0000, 1.000))]
#
# 	# Normalized to D65 https://en.wikipedia.org/wiki/LMS_color_space
# 	HPE_NORMALIZED = [(
# 		(0.4002, 0.7075, -0.0807),
# 		(-0.2280, 1.1500, 0.0612),
# 		(0.0000, 0.0000, 0.9184))]
#
# 	# CIECAM97s = np.array((
# 	#     (0.4002, 0.7075, -0.0807),
# 	#     (-0.2280, 1.1500, 0.0612),
# 	#     (0.0000, 0.0000, 0.9184)))
