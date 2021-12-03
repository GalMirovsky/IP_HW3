import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
from scipy.signal import convolve2d


def addSPnoise(im, p):

    sp_noise_im = im.copy()

    list_indices = np.arange(sp_noise_im.size)
    indices_to_change = np.array(random.sample(list_indices, round(p * list_indices.size)))
    indices_to_change_to_0 = indices_to_change[:indices_to_change.size/2]
    indices_to_change_to_255 = indices_to_change[indices_to_change.size/2:]

    sp_noise_im = np.ravel(sp_noise_im)

    sp_noise_im[indices_to_change_to_0] = 0
    sp_noise_im[indices_to_change_to_255] = 255

    sp_noise_im = np.reshape(sp_noise_im, (np.shape(im)[0], np.shape(im)[1]))
    sp_noise_im = sp_noise_im.astype(np.uint8)

    return sp_noise_im


def addGaussianNoise(im, s):

    # todo what about negative values?
    gaussian_noise_im = im.copy()
    gaussian_noise_im = gaussian_noise_im.astype(float)

    gaussian_noise = np.random.normal(loc=0, scale=s, size=(np.shape(im)[0], np.shape(im)[1]))
    gaussian_noise_im = np.add(gaussian_noise_im, gaussian_noise)

    gaussian_noise_im = gaussian_noise_im.astype(np.uint8)
    return gaussian_noise_im


def cleanImageMedian(im, radius):

    median_im = im.copy()
    im = im.astype(float)

    for ix in range(radius, im.np.shape(im)[0] - radius):
        for iy in range(radius, im.np.shape(im)[1] - radius):

            window = im[iy - radius: iy + radius + 1, ix - radius: ix + radius + 1]
            median = np.median(window)
            median_im[ix, iy] = median

    median_im = median_im.astype(np.uint8)
    return median_im


def cleanImageMean(im, radius, maskSTD):

    cleaned_im = im.copy()
    cleaned_im = cleaned_im.astype(float)

    mask = np.fromfunction(lambda x, y:  np.exp(-((np.power(x, 2) + np.power(y, 2)) / (2 * np.power(maskSTD, 2)))),
                           (-radius, radius), dtype=float)

    mask = mask / np.sum(mask)

    cleaned_im = convolve2d(cleaned_im, mask, mode='same')

    cleaned_im = cleaned_im.astype(np.uint8)
    return cleaned_im



def bilateralFilt(im, radius, stdSpatial, stdIntensity):
    bilateral_im = im.copy()

    return bilateral_im
