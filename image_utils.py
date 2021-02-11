""" ------------- Imports -------------
"""
import numpy as np
from PIL import Image

""" ------------- Functions -------------
"""


def pre_process_image(image_path):
    """ Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    """
    image = Image.open(image_path)

    # 1. Scale shortest side to 256 pixels, keep aspect ratio
    scaled_image = None
    width_idx = 0
    height_idx = 1
    smallest_dim = min(image.size[width_idx], image.size[height_idx])
    smallest_dim_idx = image.size.index(smallest_dim)

    if smallest_dim_idx == width_idx:
        resize_scalar = 256 / image.width
        scaled_image = image.resize((256, int(image.height * resize_scalar)))
    else:
        resize_scalar = 256 / image.height
        scaled_image = image.resize((int(image.width * resize_scalar), 256))

    # 2. Crop the image (center 224 X 224)
    cropped_image = scaled_image.crop(box=(scaled_image.width / 2 - 112,
                                           scaled_image.height / 2 - 112,
                                           scaled_image.width / 2 + 112,
                                           scaled_image.height / 2 + 112))

    # 3. Normalize color channels
    means = np.array([0.485, 0.456, 0.406])
    standard_deviations = np.array([0.229, 0.224, 0.225])

    image_array = np.array(cropped_image) / 255.0
    normalized_image_array = (image_array - means) / standard_deviations

    # 4. Transpose to match Pytorch's img format (color channel in the first dimension)
    transposed_image_array = normalized_image_array.transpose((2, 0, 1))

    return transposed_image_array
