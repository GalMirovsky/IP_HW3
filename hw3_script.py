import cv2.cv2 as cv2
import matplotlib.pyplot as plt

from hw3_functions import *


def main():
    print_IDs()

    lena = cv2.imread(r"Images\lena.tif")
    lena_gray = cv2.cvtColor(lena, cv2.COLOR_BGR2GRAY)

    # 1 ----------------------------------------------------------
    # add salt and pepper noise - low
    noise_name = 'salt and pepper - low'
    low_SP_rate = 0.05
    lena_sp_low = addSPnoise(lena_gray, low_SP_rate)

    # median_radius = 2
    # median_clean = cleanImageMedian(lena_sp_low, median_radius)
    # cv2.imwrite("salt_and_pepper_low_median.jpg", median_clean)
    median_clean_im = cv2.imread("salt_and_pepper_low_median.jpg")

    # mean_radius = 2
    # mean_mask_std = 2
    # mean_clean = cleanImageMean(lena_sp_low, mean_radius, mean_mask_std)
    # cv2.imwrite("salt_and_pepper_low_mean.jpg", mean_clean)
    mean_clean_im = cv2.imread("salt_and_pepper_low_mean.jpg")

    # bi_radius = 4
    # bi_spatial_std = 30
    # bi_intensity_std = 45
    # blf = bilateralFilt(lena_sp_low, bi_radius, bi_spatial_std, bi_intensity_std)
    # cv2.imwrite("salt_and_pepper_low_blf.jpg", blf)
    blf_im = cv2.imread("salt_and_pepper_low_blf.jpg")

    plot_results(lena_gray, lena_sp_low, noise_name, median_clean_im, mean_clean_im, blf_im)

    print(
        f"Conclusions for {noise_name} noise -----\n"
        f"1. Median because the S&P noise was cleaned and the image is still sharp.\n"
        f"2. Mean because the S&P noise was cleaned (less than the Median filter), but the cleaned image is blurred."
        f" Because it affects non-noise pixels.\n"
        f"3. Bilateral because the S&P noise almost was not cleaned. Happens because the filter thinks the noise"
        f" is an edge.\n"
    )

    # 2 ----------------------------------------------------------
    # add salt and pepper noise - high
    noise_name = 'salt and pepper - high'
    high_SP_rate = 0.4
    lena_sp_high = addSPnoise(lena_gray, high_SP_rate)

    # median_radius = 2
    # median_clean = cleanImageMedian(lena_sp_high, median_radius)
    # cv2.imwrite("salt_and_pepper_high_median.jpg", median_clean)
    median_clean_im = cv2.imread("salt_and_pepper_high_median.jpg")

    # mean_radius = 2
    # mean_mask_std = 2
    # mean_clean = cleanImageMean(lena_sp_high, mean_radius, mean_mask_std)
    # cv2.imwrite("salt_and_pepper_high_mean.jpg", mean_clean)
    mean_clean_im = cv2.imread("salt_and_pepper_high_mean.jpg")

    # bi_radius = 4
    # bi_spatial_std = 30
    # bi_intensity_std = 45
    # blf = bilateralFilt(lena_sp_high, bi_radius, bi_spatial_std, bi_intensity_std)
    # cv2.imwrite("salt_and_pepper_high_blf.jpg", blf)
    blf_im = cv2.imread("salt_and_pepper_high_blf.jpg")

    plot_results(lena_gray, lena_sp_high, noise_name, median_clean_im, mean_clean_im, blf_im)

    print(
        f"Conclusions for {noise_name} noise -----\n"
        f"1. Median because the S&P noise was cleaned and the image is still sharp.\n"
        f"2. Mean because the S&P noise was partially cleaned (less than the Median filter),"
        f" but the image is very blurred because it affects non-noise pixels.\n"
        f"3. Bilateral because the S&P noise was not cleaned at all and the image is very blurry."
        f" Happens because the filter thinks the noise is an edge.\n"
    )

    # 3 ----------------------------------------------------------
    # add gaussian noise - low
    noise_name = 'gaussian noise - low'
    low_gaussian_std = 20
    lena_gaussian = addGaussianNoise(lena_gray, low_gaussian_std)

    # median_radius = 3
    # median_clean = cleanImageMedian(lena_gaussian, median_radius)
    # cv2.imwrite("gaussian_noise_low_median.jpg", median_clean)
    median_clean_im = cv2.imread("gaussian_noise_low_median.jpg")

    # mean_radius = 3
    # mean_mask_std = 2
    # mean_clean = cleanImageMean(lena_gaussian, mean_radius, mean_mask_std)
    # cv2.imwrite("gaussian_noise_low_mean.jpg", mean_clean)
    mean_clean_im = cv2.imread("gaussian_noise_low_mean.jpg")

    # bi_radius = 4
    # bi_spatial_std = 15
    # bi_intensity_std = 30
    # blf = bilateralFilt(lena_gaussian, bi_radius, bi_spatial_std, bi_intensity_std)
    # cv2.imwrite("gaussian_noise_low_blf.jpg", blf)
    blf_im = cv2.imread("gaussian_noise_low_blf.jpg")

    plot_results(lena_gray, lena_gaussian, noise_name, median_clean_im, mean_clean_im, blf_im)

    print(
        f"Conclusions for {noise_name} noise -----\n"
        f"1. Bilateral because the gaussian noise was almost cleaned.\n"
        f"2. Mean because the image was cleaned (less than the Bilateral filter) but the image is blurred"
        f" because it affects non-noise pixels.\n"
        f"3. Median because the cleaned image is blurred (more than the Mean filter)."
        f" Happens because gaussian noise values are very close to the image values.\n"
    )

    # 4 ----------------------------------------------------------
    # add gaussian noise - high
    noise_name = 'gaussian noise - high'
    high_gaussian_std = 55
    lena_gaussian = addGaussianNoise(lena_gray, high_gaussian_std)

    # median_radius = 2
    # median_clean = cleanImageMedian(lena_gaussian, median_radius)
    # cv2.imwrite("gaussian_noise_high_median.jpg", median_clean)
    median_clean_im = cv2.imread("gaussian_noise_high_median.jpg")

    # mean_radius = 2
    # mean_mask_std = 5
    # mean_clean = cleanImageMean(lena_gaussian, mean_radius, mean_mask_std)
    # cv2.imwrite("gaussian_noise_high_mean.jpg", mean_clean)
    mean_clean_im = cv2.imread("gaussian_noise_high_mean.jpg")

    # bi_radius = 4
    # bi_spatial_std = 30
    # bi_intensity_std = 90
    # blf = bilateralFilt(lena_gaussian, bi_radius, bi_spatial_std, bi_intensity_std)
    # cv2.imwrite("gaussian_noise_high_blf.jpg", blf)
    blf_im = cv2.imread("gaussian_noise_high_blf.jpg")

    plot_results(lena_gray, lena_gaussian, noise_name, median_clean_im, mean_clean_im, blf_im)

    print(
        f"Conclusions for {noise_name} noise -----\n"
        f"1. Bilateral because the gaussian noise was almost cleaned.\n"
        f"2. Mean because the image was partially cleaned (less than the Bilateral filter) but the image is very blurry"
        f" because it affects non-noise pixels.\n"
        f"3. Median because the image is very blurred (more than the Mean filter) and the noise was not cleaned."
        f" Happens because gaussian noise values are very close to the image values.\n"
    )

    plt.show()


def plot_results(original, noisy, noise_name, median_clean, mean_clean, bi_f):
    plt.figure()
    plt.subplot(2, 3, 1)
    plt.imshow(original, cmap='gray', vmin=0, vmax=255)
    plt.title("original")
    plt.subplot(2, 3, 2)
    plt.imshow(noisy, cmap='gray', vmin=0, vmax=255)
    plt.title(noise_name)
    plt.subplot(2, 3, 4)
    plt.imshow(median_clean, cmap='gray', vmin=0, vmax=255)
    plt.title("median")
    plt.subplot(2, 3, 5)
    plt.imshow(mean_clean, cmap='gray', vmin=0, vmax=255)
    plt.title("mean")
    plt.subplot(2, 3, 6)
    plt.imshow(bi_f, cmap='gray', vmin=0, vmax=255)
    plt.title("bilateral")


if __name__ == '__main__':
    main()
