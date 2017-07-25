import cv2
import numpy as np
import math


def rgb_to_yiq(src, dst=None):
    if dst is None:
        dst = np.zeros((src.shape[0], src.shape[1], 3), np.float32)

    # Y
    dst[:, :, 0] = (0.1140 * src[:, :, 0] + 0.587 * src[:, :, 1] + 0.2999 * src[:, :, 2]) / 255.0
    # I
    dst[:, :, 1] = (-0.3213 * src[:, :, 0] - 0.2745 * src[:, :, 1] + 0.5957 * src[:, :, 2]) / 255.0
    # Q
    dst[:, :, 2] = (0.3114 * src[:, :, 0] - 0.5226 * src[:, :, 1] + 0.2115 * src[:, :, 2]) / 255.0

    return dst


def rgb_to_yuv(src, dst=None):
    if dst is None:
        dst = np.zeros((src.shape[0], src.shape[1], 3), np.uint8)
    # Y
    dst[:, :, 0] = (25.0 * src[:, :, 0] + 129.0 * src[:, :, 1] + 66.0 * src[:, :, 2] + 128) / 256 + 16
    # U
    dst[:, :, 1] = (112.0 * src[:, :, 0] - 74.0 * src[:, :, 1] - 38.0 * src[:, :, 2] + 128) / 256 + 128
    # V
    dst[:, :, 2] = (-18.0 * src[:, :, 0] - 94.0 * src[:, :, 1] + 112.0 * src[:, :, 2] + 128) / 256 + 128

    return dst


def rgb_to_lab(src, dst=None):
    if dst is None:
        dst = np.zeros((src.shape[0], src.shape[1], 3), np.float32)
    # L
    dst[:, :, 0] = 0.556 * src[:, :, 0] + 0.8265 * src[:, :, 1] + 0.3475 * src[:, :, 2]
    # A
    dst[:, :, 1] = -0.6411 * src[:, :, 0] + 0.4266 * src[:, :, 1] + 0.2162 * src[:, :, 2]
    # B
    dst[:, :, 2] = -0.0269 * src[:, :, 0] - 0.1033 * src[:, :, 1] + 0.1304 * src[:, :, 2]

    return dst


def rgb_to_lms(src, dst=None):
    if dst is None:
        dst = np.zeros((src.shape[0], src.shape[1], 3), np.float32)

    # L
    dst[:, :, 0] = 0.0402 * src[:, :, 0] + 0.5783 * src[:, :, 1] + 0.3811 * src[:, :, 2]
    # M
    dst[:, :, 1] = 0.0782 * src[:, :, 0] + 0.7244 * src[:, :, 1] + 0.1967 * src[:, :, 2]
    # S
    dst[:, :, 2] = 0.8444 * src[:, :, 0] + 0.1288 * src[:, :, 1] + 0.0241 * src[:, :, 2]

    return dst


def yiq_to_rgb(src, dst=None):
    if dst is None:
        dst = np.zeros((src.shape[0], src.shape[1], 3), np.uint8)
    # B
    dst[:, :, 0] = np.clip((src[:, :, 0] - 1.107 * src[:, :, 1] + 1.7046 * src[:, :, 2])*255, 0, 255)
    # G
    dst[:, :, 1] = np.clip((src[:, :, 0] - 0.2721 * src[:, :, 1] - 0.6474 * src[:, :, 2])*255, 0, 255)
    # R
    dst[:, :, 2] = np.clip((src[:, :, 0] + 0.9563 * src[:, :, 1] + 0.6210 * src[:, :, 2])*255, 0, 255)

    return dst


def yuv_to_rgb(src, dst=None):
    if dst is None:
        dst = np.zeros((src.shape[0], src.shape[1], 3), np.uint8)
    C = src[:, :, 0].astype(float) - 16
    D = src[:, :, 1].astype(float) - 128
    E = src[:, :, 2].astype(float) - 128
    # B
    dst[:, :, 0] = np.clip((298 * C + 516 * D + 128) / 256, 0, 255)
    # G
    dst[:, :, 1] = np.clip((298 * C - 100 * D - 208 * E + 128) / 256, 0, 255)
    # R
    dst[:, :, 2] = np.clip((298 * C + 409 * E + 128) / 256, 0, 255)

    return dst


def lab_to_rgb(src, dst=None):
    if dst is None:
        dst = np.zeros((src.shape[0], src.shape[1], 3), np.uint8)
    # B
    dst[:, :, 0] = np.clip(0.58841 * src[:, :, 0] - 1.0628 * src[:, :, 1] + 0.2076 * src[:, :, 2], 0, 255)
    # G
    dst[:, :, 1] = np.clip(0.5701 * src[:, :, 0] + 0.6072 * src[:, :, 1] - 2.5452 * src[:, :, 2], 0, 255)
    # R
    dst[:, :, 2] = np.clip(0.5881 * src[:, :, 0] + 0.2621 * src[:, :, 1] + 5.6958 * src[:, :, 2], 0, 255)

    return dst


def lms_to_lab(src, dst=None):
    if dst is None:
        dst = np.zeros((src.shape[0], src.shape[1], 3), np.float32)

    # l
    dst[:, :, 0] = 0.5774 * src[:, :, 0] + 0.5774 * src[:, :, 1] + 0.5774 * src[:, :, 2]
    # a
    dst[:, :, 1] = 0.4083 * src[:, :, 0] + 0.4083 * src[:, :, 1] - 0.8165 * src[:, :, 2]
    # b
    dst[:, :, 2] = 0.7071 * src[:, :, 0] - 0.7071 * src[:, :, 1]

    return dst


def lab_to_lms(src, dst=None):
    if dst is None:
        dst = np.zeros((src.shape[0], src.shape[1], 3), np.float32)

    # L
    dst[:, :, 0] = 0.5774 * src[:, :, 0] + 0.4083 * src[:, :, 1] + 0.7071 * src[:, :, 2]
    # M
    dst[:, :, 1] = 0.5774 * src[:, :, 0] + 0.4083 * src[:, :, 1] - 0.7071 * src[:, :, 2]
    # S
    dst[:, :, 2] = 0.5774 * src[:, :, 0] - 0.8165 * src[:, :, 1]

    return dst


def lms_to_rgb(src, dst=None):
    if dst is None:
        dst = np.zeros((src.shape[0], src.shape[1], 3), np.uint8)

    # B
    dst[:, :, 0] = np.clip(0.0497 * src[:, :, 0] - 0.2439 * src[:, :, 1] + 1.2045 * src[:, :, 2], 0, 255)
    # G
    dst[:, :, 1] = np.clip(-1.2186 * src[:, :, 0] + 2.3809 * src[:, :, 1] - 0.1624 * src[:, :, 2], 0, 255)
    # R
    dst[:, :, 2] = np.clip(4.4679 * src[:, :, 0] - 3.5873 * src[:, :, 1] + 0.1193 * src[:, :, 2], 0, 255)

    return dst


def log(src, dst=None):
    if dst is None:
        dst = np.zeros((src.shape[0], src.shape[1], 3), np.float32)

    for y in range(src.shape[0]):
        for x in range(src.shape[1]):
            dst[y, x, 0] = math.log10(src[y, x, 0])
            dst[y, x, 1] = math.log10(src[y, x, 1])
            dst[y, x, 2] = math.log10(src[y, x, 2])

    return dst


def power10(src, dst=None):
    if dst is None:
        dst = np.zeros((src.shape[0], src.shape[1], 3), np.float32)

    for y in range(src.shape[0]):
        for x in range(src.shape[1]):
            dst[y, x, 0] = math.pow(10.0, src[y, x, 0])
            dst[y, x, 1] = math.pow(10.0, src[y, x, 1])
            dst[y, x, 2] = math.pow(10.0, src[y, x, 2])

    return dst


def color_transfer(src, target, dst=None, log_space=False):
    if dst is None:
        dst = np.zeros((target.shape[0], target.shape[1], 3), np.float32)

    if log_space:
        src_lms = rgb_to_lms(src)
        target_lms = rgb_to_lms(target)

        src_lms = log(src_lms)
        target_lms = log(target_lms)

        src_lab = lms_to_lab(src_lms)
        target_lab = lms_to_lab(target_lms)

    else:
        # Transform to lab color space
        src_lab = rgb_to_lab(src)
        target_lab = rgb_to_lab(target)

    src_mean = np.zeros(3)
    src_mean[0] = np.mean(src_lab[:, :, 0])
    src_mean[1] = np.mean(src_lab[:, :, 1])
    src_mean[2] = np.mean(src_lab[:, :, 2])

    src_std = np.zeros(3)
    src_std[0] = np.std(src_lab[:, :, 0])
    src_std[1] = np.std(src_lab[:, :, 1])
    src_std[2] = np.std(src_lab[:, :, 2])

    target_mean = np.zeros(3)
    target_mean[0] = np.mean(target_lab[:, :, 0])
    target_mean[1] = np.mean(target_lab[:, :, 1])
    target_mean[2] = np.mean(target_lab[:, :, 2])

    target_std = np.zeros(3)
    target_std[0] = np.std(target_lab[:, :, 0])
    target_std[1] = np.std(target_lab[:, :, 1])
    target_std[2] = np.std(target_lab[:, :, 2])

    dst[:, :, 0] = (target_std[0] / src_std[0]) * (src_lab[:, :, 0] - src_mean[0]) + target_mean[0]
    dst[:, :, 1] = (target_std[1] / src_std[1]) * (src_lab[:, :, 1] - src_mean[1]) + target_mean[1]
    dst[:, :, 2] = (target_std[2] / src_std[2]) * (src_lab[:, :, 2] - src_mean[2]) + target_mean[2]

    if log_space:
        dst_lms = lab_to_lms(dst)
        dst_lms = power10(dst_lms)
        dst_rgb = lms_to_rgb(dst_lms)

    else:
        dst_rgb = lab_to_rgb(dst)

    return dst_rgb


def yiq_transfer(src, target, dst=None, transfer_iq=False):
    if dst is None:
        dst = np.zeros((src.shape[0], src.shape[1], 3), np.float32)

    src_yiq = rgb_to_yiq(src)
    target_yiq = rgb_to_yiq(target)

    src_mean = np.zeros(3)
    src_mean[0] = np.mean(src_yiq[:, :, 0])

    src_std = np.ones(3)
    src_std[0] = np.std(src_yiq[:, :, 0])

    if transfer_iq:
        src_mean[1] = np.mean(src_yiq[:, :, 1])
        src_mean[2] = np.mean(src_yiq[:, :, 2])

        src_std[1] = np.std(src_yiq[:, :, 1])
        src_std[2] = np.std(src_yiq[:, :, 2])

    target_mean = np.zeros(3)
    target_mean[0] = np.mean(target_yiq[:, :, 0])

    target_std = np.ones(3)
    target_std[0] = np.std(target_yiq[:, :, 0])

    if transfer_iq:
        target_mean[1] = np.mean(target_yiq[:, :, 1])
        target_mean[2] = np.mean(target_yiq[:, :, 2])

        target_std[1] = np.std(target_yiq[:, :, 1])
        target_std[2] = np.std(target_yiq[:, :, 2])

    dst[:, :, 0] = (target_std[0] / src_std[0]) * (src_yiq[:, :, 0] - src_mean[0]) + target_mean[0]
    dst[:, :, 1] = (target_std[1] / src_std[1]) * (src_yiq[:, :, 1] - src_mean[1]) + target_mean[1]
    dst[:, :, 2] = (target_std[2] / src_std[2]) * (src_yiq[:, :, 2] - src_mean[2]) + target_mean[2]

    dst_rgb = yiq_to_rgb(dst)

    return dst_rgb


def brightness_transfer(src, target, dst=None):
    if dst is None:
        dst = np.zeros((src.shape[0], src.shape[1], 3), np.float32)

    src_yuv = rgb_to_yuv(src)
    target_yuv = rgb_to_yuv(target)

    src_mean = np.mean(src_yuv[:, :, 0])
    src_std = np.std(src_yuv[:, :, 0])

    target_mean = np.mean(target_yuv[:, :, 0])
    target_std = np.std(target_yuv[:, :, 0])

    dst[:, :, 0] = (target_std / src_std) * (src_yuv[:, :, 0] - src_mean) + target_mean
    dst[:, :, 1] = src_yuv[:, :, 1]
    dst[:, :, 2] = src_yuv[:, :, 2]

    dst_rgb = yuv_to_rgb(dst)

    return dst_rgb


def main():
    target = cv2.imread("Images/cima.png")
    src = cv2.imread("Images/baixo.png")

    dst = yiq_transfer(src, target, transfer_iq=True)

    cv2.imshow("color", dst)
    cv2.waitKey(0)


def main2():
    target = cv2.imread("Images/baixo.png")
    src = cv2.imread("Images/cima.png")

    dst = color_transfer(src, target, log_space=True)

    cv2.imshow("color", dst)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()
