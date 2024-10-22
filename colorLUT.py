import numpy as np

"""
    Black color needs to be always first. 
    In mask image we use color Lut to add color to image pixels.
    Pixels with value 0 get the black color (background)
"""

# As HSV
colorLUT = np.zeros((256, 1, 3), dtype=np.uint8)
colorLUT[:8, 0, :] = [[0, 0, 0], # black 
                     [0, 200, 255], # red
                     [30, 200, 255],  # yellow
                     [130, 200, 255],  # purple
                     [60, 200, 255], # green
                     [100, 200, 128], #cyan
                     [160, 200, 255], #pink
                     [130, 0, 130]  # Grey
                     ]  