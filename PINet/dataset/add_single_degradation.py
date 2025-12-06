from pathlib import Path
import numpy as np
import cv2
from scipy.io import loadmat
import torch
import math
from typing import Optional
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from basicsr.utils.matlab_functions import imresize

__all__ = [
    "lr", 
    "darken", 
    "add_noise", 
    "add_jpeg_comp_artifacts", 
    "add_haze",
    "add_motion_blur",
    "add_defocus_blur",
    "add_rain",
]

def lr(img, keep_size=False):

    img = img.copy()

    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1)
    img = imresize(img, scale=0.25)
    if keep_size:
        img = imresize(img, scale=4)
    img = img.permute(1, 2, 0).numpy()
    img = (img * 255).clip(0, 255).round().astype(np.uint8)

    return img

def add_noise(img, noise_type: Optional[str] = None, arg=None):

    img = img.copy()
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    types = ["Gaussian", "Poisson"]
    if noise_type is None:
        noise_type = np.random.choice(types)
    else:
        assert noise_type in types

    if noise_type == "Gaussian":
        if arg is None:
            sigma_range = [10, 20]
        else:
            sigma_range = [arg, arg]
        out = random_add_gaussian_noise_pt(
            img, 
            sigma_range=sigma_range,
            clip=True, 
            rounds=False)
        
    else:
        if arg is None:
            scale_range = [1, 1.5]
        else:
            scale_range = [arg, arg]
        out = random_add_poisson_noise_pt(
            img,
            scale_range=scale_range,
            clip=True,
            rounds=False)
        
    lq = out.squeeze(0).permute(1, 2, 0).cpu().numpy()
    lq = (lq * 255).clip(0,255).round().astype(np.uint8)
    return lq

def add_jpeg_comp_artifacts(img, quality_factor: Optional[int] = None):

    img = img.copy()
    if quality_factor is None:
        quality_factor = np.random.randint(50, 80)
    _, encimg = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality_factor])
    img = cv2.imdecode(encimg, cv2.IMREAD_COLOR)
    return img


def darken(img, darken_type: Optional[str] = None, arg=None):

    img = img.copy()

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    types = ["constant shift", "gamma correction", "linear mapping"]
    if darken_type is None:
        darken_type = np.random.choice(types)
    else:
        assert darken_type in types

    if darken_type == "constant shift":
        if arg is None:
            shift = np.random.randint(10, 30)
        else:
            shift = arg
        v = np.clip(np.int16(v)-shift, 0, 255).round().astype(np.uint8)
    elif darken_type == "gamma correction":

        if arg is None:
            gamma = np.random.uniform(0.8, 0.9)
        else:
            gamma = arg
        v = (cv2.pow(v / 255.0, 1.0 / gamma) * 255).clip(0,255).round().astype(np.uint8)
    else:

        if arg is None:
            dst_max = np.random.randint(200, 240)
        else:
            dst_max = arg
        vmin, vmax = np.min(v), np.max(v)
        v = ((v - vmin) / (vmax - vmin) * dst_max).round().astype(np.uint8)
    
    hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def add_haze(img, intensity=0.3, A=0.7):
    img = img.astype(np.float32) / 255.0

    h, w, _ = img.shape
    xx, yy = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
    depth = (xx * xx + yy * yy)
    depth = (depth - depth.min()) / (depth.max() - depth.min())

    beta = intensity * 2.0
    t = np.exp(-beta * depth)
    t = t[..., None]

    A = np.clip(A, 0, 1)

    hazy = img * t + A * (1 - t)

    hazy = (hazy * 255).clip(0, 255).astype(np.uint8)
    return hazy


def add_motion_blur(img, severity: Optional[int] = None):
    img = img.copy()

    severity = 0
    radius, sigma = [(10, 3), (15, 5), (8, 2)][severity]
    
    angle = np.random.uniform(-90, 90)

    width = radius * 2 + 1

    k = (np.exp(-np.arange(width)**2 / (2*(sigma**2)))) / (np.sqrt(2*np.pi)*sigma)  # gaussian
    kernel = k / np.sum(k)
    point = (width * np.sin(np.deg2rad(angle)), width * np.cos(np.deg2rad(angle)))
    hypot = math.hypot(point[0], point[1])

    blurred = np.zeros_like(img, dtype=np.float32)
    for i in range(width):
        dy = -math.ceil(((i*point[0]) / hypot) - 0.5)
        dx = -math.ceil(((i*point[1]) / hypot) - 0.5)
        if np.abs(dy) >= img.shape[0] or np.abs(dx) >= img.shape[1]:
            break

        if dx < 0:
            shifted = np.roll(img, shift=img.shape[1]+dx, axis=1)
            shifted[:,dx:] = shifted[:,dx-1:dx]
        elif dx > 0:
            shifted = np.roll(img, shift=dx, axis=1)
            shifted[:,:dx] = shifted[:,dx:dx+1]
        else:
            shifted = img

        if dy < 0:
            shifted = np.roll(shifted, shift=img.shape[0]+dy, axis=0)
            shifted[dy:,:] = shifted[dy-1:dy,:]
        elif dy > 0:
            shifted = np.roll(shifted, shift=dy, axis=0)
            shifted[:dy,:] = shifted[dy:dy+1,:]

        blurred = blurred + kernel[i] * shifted

    img = np.clip(blurred, 0, 255).round().astype(np.uint8)
    return img

def add_defocus_blur(img, severity: Optional[int] = None):

    img = img.copy()

    severity = 2
    radius, alias_blur = [(3, 0.1), (4, 0.5), (6, 0.5)][severity]

    if radius <= 8:
        L = np.arange(-8, 8 + 1)
        ksize = (3, 3)
    else:
        L = np.arange(-radius, radius + 1)
        ksize = (5, 5)
    X, Y = np.meshgrid(L, L)
    aliased_disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=np.float32)
    aliased_disk /= np.sum(aliased_disk)

    kernel = cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)

    img = img / 255.0
    channels = []
    for d in range(3):
        channels.append(cv2.filter2D(img[:, :, d], -1, kernel))
    channels = np.array(channels).transpose((1, 2, 0))

    img = (np.clip(channels, 0, 1) * 255).round().astype(np.uint8)
    return img


def add_rain(img, value: Optional[int] = None):
    img = img.copy()

    w = 3
    length = np.random.randint(10, 50)
    angle = np.random.randint(-30, 30)

    if value is None:
        value = np.random.randint(10, 80)
    noise = np.random.uniform(0, 256, img.shape[0:2])
    v = value * 0.01
    noise[np.where(noise < (256 - v))] = 0

    k = np.array([[0, 0.1, 0],
                  [0.1, 8, 0.1],
                  [0, 0.1, 0]])

    noise = cv2.filter2D(noise, -1, k)

    trans = cv2.getRotationMatrix2D((length / 2, length / 2), angle - 45, 1 - length / 100.0)
    dig = np.diag(np.ones(length))
    k = cv2.warpAffine(dig, trans, (length, length))
    k = cv2.GaussianBlur(k, (w, w), 0)

    blurred = cv2.filter2D(noise, -1, k)

    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)

    rain = np.expand_dims(blurred, 2)
    rain = np.repeat(rain, 3, 2)

    img = img.astype('float32') + rain
    np.clip(img, 0, 255, out=img)

    return img.round().astype(np.uint8)
