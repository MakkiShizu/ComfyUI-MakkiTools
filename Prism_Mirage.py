from .convert_tensor import tensor2pil, pil2tensor
from PIL import Image
import numpy as np


# 1:exterior
# 2:interior
def process_images(image1_path, image2_path, exterior_level, interior_level):

    img1 = tensor2pil(image1_path).convert("RGBA")
    img2 = tensor2pil(image2_path).convert("RGBA")

    img1_trans = diagonal_transparency(img1, offset=1)
    img1_adjusted = adjust_levels(img1_trans, in_black=exterior_level, out_white=255)

    img2_trans = diagonal_transparency(img2, offset=0)
    img2_adjusted = adjust_levels(img2_trans, out_white=interior_level)

    result = Image.alpha_composite(img2_adjusted, img1_adjusted)
    result = pil2tensor(result)

    return result


def diagonal_transparency(img, offset=0):
    """Create diagonal transparency effect"""
    arr = np.array(img)
    height, width, _ = arr.shape

    mask = np.zeros((height, width), dtype=bool)
    for y in range(height):
        for x in range(width):
            mask[y, x] = (x + y) % 2 == offset

    arr[mask] = [0, 0, 0, 0]

    return Image.fromarray(arr)


def adjust_levels(img, in_black=0, in_white=255, out_black=0, out_white=255):
    """Adjust the input/output color scale of the image"""
    arr = np.array(img, dtype=np.float32)

    if in_black > 0 or in_white < 255:
        in_range = in_white - in_black
        arr[..., :3] = np.clip((arr[..., :3] - in_black) * (255 / in_range), 0, 255)

    if out_black > 0 or out_white < 255:
        out_range = out_white - out_black
        arr[..., :3] = np.clip(arr[..., :3] * (out_range / 255) + out_black, 0, 255)

    return Image.fromarray(arr.astype(np.uint8))
