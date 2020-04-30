usage: white_balance.py [-h] [-p FULL_PATH] [-fp FLASH_REGIONS]
                        [-sp SHADOW_REGIONS] [-b BRIGHTNESS] [-s SATURATION]
                        [-o OUT_PATH]

White Balance from two images, flash and no flash. Given two images of
identical scenes, perform a white-balance improvement on the no-flash image,
using the flash's chromaticity. The two images must be taken using manual
settings, so that only the flash acts as a controlled difference between them.

See https://docs.google.com/document/d/1lh4-qXczKkjLZlF1Q3_6r5FPN-q3uqodqbx9Tx_DFZg/edit?usp=sharing
for algorithm details.

optional arguments:
  -h, --help            show this help message and exit
  -p FULL_PATH, --path FULL_PATH
                        Path must be either the name of dir or path to one of
                        the images. If image file path is provided, provide
                        one image path such as im1_noflash.JPG. The other two
                        images will follow the same format -
                        <name><suffix>.<extension>. If path to directory is
                        provided, supply a directory with three images
                        containing the default names: noflash.tiff,
                        withflash.tiff, graycard.tiff
  -fp FLASH_REGIONS, --flash FLASH_REGIONS
                        Optional control over estimated spectral flash
                        reflections' percentage of pixels. Defaults to 0.005,
                        or 0.5%. Takes values from 0 (No reflections caused by
                        the cameras flash, such as a open-space scene or non-
                        reflective materials) to 1 (The entire image is a
                        flash reflection. Not very likely.) Flash regions are
                        normally small, therefore an input of less than 1%, or
                        0.01 is reasonable.
  -sp SHADOW_REGIONS, --shadow SHADOW_REGIONS
                        Optional control over estimated shadow percentage of
                        pixels. Defaults to 0.25, or 25.0%. Takes values from
                        0 (No shadows caused by the cameras flash, such as a
                        open-space scene with no reflective walls) to 1 (The
                        entire image is a shadow caused by the flash. Not very
                        likely.) Shadowy regions are normally large, therefore
                        an input of about 20%, or 0.2 is reasonable for a
                        portrait in confined space, where the camera's flash
                        is likely to cause shadows.
  -b BRIGHTNESS, --brightness BRIGHTNESS
                        Optional control over brightness. Defaults to 0.5.
                        Takes values from 0 (dark) to 1 (light), with input of
                        0.5 for no effect. Note that a balance between this
                        parameter and saturation can lead to better results
                        and obscure errors.
  -s SATURATION, --saturation SATURATION
                        Optional control over saturation. Defaults to 0.65.
                        Takes values from 0 (pale) to 1 (vivid), with input of
                        0.5 for no effect. Note that a balance between this
                        parameter and brightness can lead to better results
                        and obscure errors. How strong should the colors come
                        through after white balance correction?
  -o OUT_PATH, --out_path OUT_PATH
                        Optional. Path in which to save result. If not
                        specified, the result will be displayed instead. Two
                        images are saves, one without post-processing
                        <out_path> and one gamma corrected (brighter)
                        <out_dir> + gamma_corrected_ <out_basename>