import numpy as np
import cv2
import random
from PIL import Image

def add_gaussian_noise(img, mean=0, std=50):
    h, w, _ = img.shape
    patch_h, patch_w = h // 3, w // 3
    x1 = random.randint(0, w - patch_w)
    y1 = random.randint(0, h - patch_h)
    x2 = random.randint(0, w - patch_w)
    y2 = random.randint(0, h - patch_h)

    noise = np.random.normal(mean, std, (patch_h, patch_w, 3)).astype(np.uint8)
    noisy_image = img.copy()

    # Add noise to the first patch
    noisy_image[y1:y1 + patch_h, x1:x1 + patch_w] = np.clip(
        noisy_image[y1:y1 + patch_h, x1:x1 + patch_w] + noise, 0, 255).astype(np.uint8)
    
    noise = np.random.normal(mean, std, (patch_h, patch_w, 3)).astype(np.uint8)

    # Add noise to the second patch
    noisy_image[y2:y2 + patch_h, x2:x2 + patch_w] = np.clip(
        noisy_image[y2:y2 + patch_h, x2:x2 + patch_w] + noise, 0, 255).astype(np.uint8)
    
    return noisy_image

def add_random_mask(img,mask_size=128):
    h,w,_=img.shape
    x = random.randint(0,w-mask_size)
    y = random.randint(0,h-mask_size)
    img[y:y+mask_size,x:x+mask_size]=0
    return img


def fast_blur_patch(img, kernel_size=15):
    """
    Applies a fast box blur only to a specified patch of the image.

    Parameters:
        img         : Input image (numpy array).
        x, y        : Top-left corner of the patch to blur.
        patch_w     : Width of the patch.
        patch_h     : Height of the patch.
        kernel_size : Kernel size for the blur.

    Returns:
        Image with the specified patch blurred.
    """

    h, w, _ = img.shape
    patch_h, patch_w = h // 3, w // 3
    x = 0
    y = 0

    img_blur = img.copy()
    roi = img_blur[y:h, x:w]
    blurred_roi = cv2.blur(roi, (kernel_size, kernel_size))
    img_blur[y:h, x:w] = blurred_roi
    return img_blur


def add_banding(img, bands=10):

    l, b, _ = img.shape
    h, w = l // 5, b // 5
    x = random.randint(0, b - w)
    y = random.randint(0, l - h)

    """
    Adds a banding effect to a rectangular region of the image.

    Parameters:
        img   : Input image (numpy array).
        x, y  : Top-left corner of the region to add banding.
        w, h  : Width and height of the region.
        bands : Number of bands to create in the region.

    Returns:
        Image with the specified region banded.
    """
    img_banded = img.copy()
    roi = img_banded[y:y+h, x:x+w]
    band_height = h // bands
    for i in range(bands):
        start = i * band_height
        end = start + band_height if i < bands - 1 else h
        # Calculate the average color of the band
        avg_color = roi[start:end, :, :].mean(axis=(0, 1), keepdims=True)
        # Assign the average color to the band
        roi[start:end, :, :] = avg_color.astype(np.uint8)
    img_banded[y:y+h, x:x+w] = roi
    return img_banded


def corrupt_image(img: Image.Image, mode: str = 'noise+mask'):
    img_np = np.array(img)

    if mode == 'noise':
        img_np = add_gaussian_noise(img_np)
    elif mode == 'mask':
        img_np = add_random_mask(img_np)
    elif mode == 'banding':
        img_np = add_banding(img_np)
    elif mode == 'All':
        img_np = add_banding(img_np)
        img_np = fast_blur_patch(img_np)
        img_np = add_gaussian_noise(img_np)
        img_np = add_random_mask(img_np)
        
    
    return Image.fromarray(img_np)