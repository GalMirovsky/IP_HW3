import cv2.cv2
import matplotlib.pyplot as plt

from hw3_functions import *


def main():
    # feel free to load different image than lena
    lena = cv2.imread(r"Images\lena.tif")
    lena_gray = cv2.cvtColor(lena, cv2.COLOR_BGR2GRAY)

    # 1 ----------------------------------------------------------
    # add salt and pepper noise - low
    noise_name = 'salt and pepper - low'
    low_SP_rate = 0.1
    lena_sp_low = addSPnoise(lena_gray, low_SP_rate)

    median_radius = 2
    median_clean = cleanImageMedian(lena_sp_low, median_radius)

    mean_radius = 10
    mean_mask_std = 5
    mean_clean = cleanImageMean(lena_sp_low, mean_radius, mean_mask_std)

    bi_radius = 3
    bi_spatial_std = 1
    bi_intensity_std = 1
    bi_f = bilateralFilt(lena_sp_low, bi_radius, bi_spatial_std, bi_intensity_std)

    plot_results(lena_gray, lena_sp_low, noise_name, median_clean, mean_clean, bi_f)
    plt.show()
    print(
        f"Conclusions for {noise_name} noise -----\n"
        f"1. "
        f"2. "
        f"3. "
    )  # TODO: add explanation

    # 2 ----------------------------------------------------------
    # add salt and pepper noise - high
    noise_name = 'salt and pepper - high'
    high_SP_rate = 0.5
    lena_sp_high = addSPnoise(lena_gray, high_SP_rate)

    median_radius = 3
    median_clean = cleanImageMedian(lena_sp_high, median_radius)

    mean_radius = 3
    mean_mask_std = 1
    mean_clean = cleanImageMean(lena_sp_high, mean_radius, mean_mask_std)

    bi_radius = 3
    bi_spatial_std = 1
    bi_intensity_std = 1
    bi_f = bilateralFilt(lena_sp_high, bi_radius, bi_spatial_std, bi_intensity_std)

    plot_results(lena_gray, lena_sp_high, noise_name, median_clean, mean_clean, bi_f)

    print(
        f"Conclusions for {noise_name} noise -----\n"
        f"1. "
        f"2. "
        f"3. "
    )  # TODO: add explanation

    # 3 ----------------------------------------------------------
    # add gaussian noise - low
    noise_name = 'gaussian noise - low'
    low_gaussian_std = 1
    lena_gaussian = addGaussianNoise(lena_gray, low_gaussian_std)

    median_radius = 3
    median_clean = cleanImageMedian(lena_gaussian, median_radius)

    mean_radius = 3
    mean_mask_std = 1
    mean_clean = cleanImageMean(lena_gaussian, mean_radius, mean_mask_std)

    bi_radius = 3
    bi_spatial_std = 1
    bi_intensity_std = 1
    bi_f = bilateralFilt(lena_gaussian, bi_radius, bi_spatial_std, bi_intensity_std)

    plot_results(lena_gray, lena_gaussian, noise_name, median_clean, mean_clean, bi_f)

    print(
        f"Conclusions for {noise_name} noise -----\n"
        f"1. "
        f"2. "
        f"3. "
    )  # TODO: add explanation

    # 4 ----------------------------------------------------------
    # add gaussian noise - high
    noise_name = 'gaussian noise - high'
    high_gaussian_std = 5
    lena_gaussian = addGaussianNoise(lena_gray, high_gaussian_std)

    median_radius = 3
    median_clean = cleanImageMedian(lena_gaussian, median_radius)

    mean_radius = 3
    mean_mask_std = 1
    mean_clean = cleanImageMean(lena_gaussian, mean_radius, mean_mask_std)

    bi_radius = 3
    bi_spatial_std = 1
    bi_intensity_std = 1
    bi_f = bilateralFilt(lena_gaussian, bi_radius, bi_spatial_std, bi_intensity_std)

    plot_results(lena_gray, lena_gaussian, noise_name, median_clean, mean_clean, bi_f)

    print(
        f"Conclusions for {noise_name} noise -----\n"
        f"1. "
        f"2. "
        f"3. "
    )  # TODO: add explanation

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
    # remove from comment when last function will be ready
    # plt.subplot(2, 3, 6)
    # plt.imshow(bi_f, cmap='gray', vmin=0, vmax=255)


if __name__ == '__main__':
    main()
