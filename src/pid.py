import numpy as np
import cv2
import random
from PIL import Image, ImageOps

CROP_SIZE = 256


def _prepare_crop(img_pil, size=CROP_SIZE, train=True):
    """
    EDIT: replaces the old `.resize((256,256), resample=LANCZOS)` call.

    Any resize that uses an interpolation kernel (LANCZOS, BICUBIC, etc.)
    mixes neighboring pixel values together. PiD's signal lives in the
    sub-pixel quantization residual, so blending pixels during a resize
    contaminates exactly the thing we're trying to measure.

    Cropping instead takes pixels verbatim -- no resampling math touches
    the values -- so the residual signal stays intact. We only lose some
    spatial coverage of the image, not signal fidelity.

    train=True  -> random crop location (acts as data augmentation, and
                   means the model sees a different patch of the same
                   image across epochs)
    train=False -> deterministic center crop (needed so validation/testing
                   results are reproducible run to run)

    Images smaller than `size` in either dimension are zero-padded first
    (a pad is NOT a resample -- it doesn't invent new pixel values, it
    just adds a black border) so the crop always has something to take.
    """
    w, h = img_pil.size

    if w < size or h < size:
        pad_w = max(size - w, 0)
        pad_h = max(size - h, 0)
        left_pad = pad_w // 2
        top_pad = pad_h // 2
        img_pil = ImageOps.expand(
            img_pil,
            border=(left_pad, top_pad, pad_w - left_pad, pad_h - top_pad),
            fill=0,
        )
        w, h = img_pil.size

    if train:
        left = random.randint(0, w - size)
        top = random.randint(0, h - size)
    else:
        left = (w - size) // 2
        top = (h - size) // 2

    return img_pil.crop((left, top, left + size, top + size))


def apply_pid_algorithm(img_pil, train=True):
    """
    this is the main implementation of the PiD algorithm used across all files for (training , inference and testing)

    EDIT: added `train` flag. Pass train=True from the training loop
    (random crop / augmentation) and train=False from validation, testing,
    and inference (deterministic center crop). See _prepare_crop above.
    """
    img_pil = _prepare_crop(img_pil, size=CROP_SIZE, train=train)  # EDIT: crop, not resize

    img = np.array(img_pil)  # convert to a numpy array to carry out maths and matrix operations
    # here i ensure that the loaded image is in rgb format (some images are in grayscale or rgba format that would mess our weight as the resnet is trained on rgb images)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    img_float = img.astype(np.float32)  # ensure we are working with a float
    # standard transforming matrix  RBG --> YUV
    Mt = np.array([
        [0.299, 0.587, 0.114],
        [-0.168736, -0.331264, 0.5],
        [0.5, -0.418688, -0.081312]
    ])
    pixels = img_float.reshape(-1, 3)  # flattening out the image to do matrices maths
    yuv_pixels = np.dot(pixels, Mt.T)  # here come the maths ( we move the image to the yuv domain by multiplying with the transform matrix)
    yuv_quantized = np.floor(yuv_pixels)  # the quantization function Q(x) to cause the precision loss that leads to the residual
    # standard inverse transforming matrix  YUV --> RGB
    Mt_inv = np.array([
        [1.0, 0.0, 1.402],
        [1.0, -0.344136, -0.714136],
        [1.0, 1.772, 0.0]
    ])
    recovered_pixels = np.dot(yuv_quantized, Mt_inv.T)  # the inverse maths (we get back to the rgb domain by multiplying with the inverse transform matrix)
    _recovered_pixels = np.round(recovered_pixels)  # another quantization to cause further precision loss
    clipped_pixels = _recovered_pixels.astype(np.float32)
    img_altered = clipped_pixels.reshape(img.shape)  # reshaping back to the original dimensions
    residual = img_float - img_altered  # get the residual

    return residual
