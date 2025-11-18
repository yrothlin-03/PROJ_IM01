from kernel_estimation_bis import estime_noyau, centrer_le_noyau
from kernel_estimation import blur_kernel_estimation
from tv_deconv import tv_deconv
from tv_deconv_bis import TVdeconv as TV_deconv
from skimage.restoration import richardson_lucy
from skimage import img_as_ubyte
from utils import *
import numpy as np
from scipy.signal import convolve2d
from time import time
import matplotlib.pyplot as plt


plt.ion()   




my_tv = False

add_tapping = True

kernel_estimation = blur_kernel_estimation

apply_motion_blur = False



" ------------------------- TV Deconvolution Test ------------------   "
" ------------------------- TV Deconvolution Test ------------------   "
" ------------------------- TV Deconvolution Test ------------------   "
" ------------------------- TV Deconvolution Test ------------------   "
" ------------------------- TV Deconvolution Test ------------------   "

def test_tv_deconv():
    images = load_test_data()
    for u in images:
        if u.ndim == 3 and u.shape[2] == 3:                             # RGB image
            u_ycbcr = rgb_to_ycbcr(u)
            u_y = u_ycbcr[:, :, 0]
            h = motion_blur_kernel(25, 90)
            v = convolve2d(u_y, h, mode='same', boundary='symm')
            t1 = time()
            if my_tv:
                u_rec, _ = tv_deconv(v, h, add_tapping=add_tapping)
            else:
                u_rec, _ = TV_deconv(v, h, 2000/255)
            t2 = time()
            dt = t2 - t1
            print(f"TV Deconvolution took {dt:.2f} seconds.")

            u_rec_ycbcr = np.stack([u_rec, u_ycbcr[:, :, 1], u_ycbcr[:, :, 2]], axis=-1)
            u_rec_rgb = ycbcr_to_rgb(u_rec_ycbcr)

            metrics = compute_metrics(u, u_rec_rgb)
            print(f"PSNR: {metrics['PSNR']:.2f} | SNR: {metrics['SNR']:.2f} | SSIM: {metrics['SSIM']:.4f}")
            u_rec = u_rec_rgb

        else:                                                           # Grayscale image       
            h = motion_blur_kernel(25, 90)
            v = convolve2d(u, h, mode='same', boundary='symm')
            t1 = time()
            if my_tv:
                u_rec, _ = tv_deconv(v, h, add_tapping=add_tapping)
            else:
                u_rec, _ = TV_deconv(v, h, 2000/255)
            t2 = time()
            dt = t2 - t1
            print(f"TV Deconvolution took {dt:.2f} seconds.")

            metrics = compute_metrics(u, u_rec)
            print(f"PSNR: {metrics['PSNR']:.2f} | SNR: {metrics['SNR']:.2f} | SSIM: {metrics['SSIM']:.4f}")


        images = [u, v, h, u_rec]
        titles = ["Original Image", "Blurred Image", "Blur Kernel", f"Deconvolved Image (PSNR: {metrics['PSNR']:.2f})"]
        view_images(*images, titles=titles, cols=4)
        plt.pause(np.inf)






" ------------------------- Noise Test ------------------ "
" ------------------------- Noise Test ------------------ "
" ------------------------- Noise Test ------------------ "
" ------------------------- Noise Test ------------------ "
" ------------------------- Noise Test ------------------ "

def noise_test():
    images = load_test_data()
    noise_levels = [0, 0.1, 1]

    for std in noise_levels:
        for u_clean in images:

            u = add_noise(u_clean, mean=0, std=std)

            if u.ndim == 3 and u.shape[2] == 3:  # RGB image
                u_ycbcr = rgb_to_ycbcr(u)
                u_y = u_ycbcr[:, :, 0]

                if apply_motion_blur:
                    h_true = motion_blur_kernel(25, 30)
                    v = convolve2d(u_y, h_true, mode="same", boundary="symm")
                else:
                    v = u_y

                t1 = time()
                h_rec = estime_noyau(v, p=25, verbose=False)[0]
                t2 = time()
                dt = t2 - t1
                print(f"Kernel estimation took {dt:.2f} seconds.")

                h_rec = centrer_le_noyau(h_rec)

                t1 = time()
                u_rec, _ = tv_deconv(v, h_rec, add_tapping=add_tapping)
                t2 = time()
                dt = t2 - t1
                print(f"TV Deconvolution took {dt:.2f} seconds.")

                u_rec_ycbcr = np.stack(
                    [u_rec, u_ycbcr[:, :, 1], u_ycbcr[:, :, 2]], axis=-1
                )
                u_rec_rgb = ycbcr_to_rgb(u_rec_ycbcr)

                metrics = compute_metrics(u_clean, u_rec_rgb)
                print(
                    f"Noise std: {std} | PSNR: {metrics['PSNR']:.2f} | "
                    f"SNR: {metrics['SNR']:.2f} | SSIM: {metrics['SSIM']:.4f}"
                )

                imgs_to_show = [u_clean, v, h_rec, u_rec_rgb]

            else:  # Grayscale image
                if apply_motion_blur:
                    h_true = motion_blur_kernel(25, 30)
                    v = convolve2d(u, h_true, mode="same", boundary="symm")
                else:
                    v = u

                t1 = time()
                h_rec = estime_noyau(v, p=25, verbose=False)[0]
                t2 = time()
                dt = t2 - t1
                print(f"Kernel estimation took {dt:.2f} seconds.")

                h_rec = centrer_le_noyau(h_rec)

                t1 = time()
                u_rec, _ = tv_deconv(v, h_rec, add_tapping=add_tapping)
                t2 = time()
                dt = t2 - t1
                print(f"TV Deconvolution took {dt:.2f} seconds.")

                metrics = compute_metrics(u_clean, u_rec)
                print(
                    f"Noise std: {std} | PSNR: {metrics['PSNR']:.2f} | "
                    f"SNR: {metrics['SNR']:.2f} | SSIM: {metrics['SSIM']:.4f}"
                )

                imgs_to_show = [u_clean, v, h_rec, u_rec]

            titles = [
                "Original Image",
                f"Blurred Noisy Image (std={std})",
                "Estimated Kernel",
                f"Deconvolved Image (PSNR: {metrics['PSNR']:.2f})",
            ]
            view_images(*imgs_to_show, titles=titles, cols=4)






def kernel_estimation_test():
    img = load_test_data()
    K = [motion_blur_kernel(25, k) for k in [0, 45, 90]]
    K.append(gaussian_kernel(25, 5))
    for h in K:
        for im in img:
            if im.ndim == 3 and im.shape[2] == 3:
                im_ycbcr = rgb_to_ycbcr(im)
                im_y = im_ycbcr[:, :, 0]
                v = convolve2d(im_y, h, mode='same', boundary='symm')
                h_est = estime_noyau(v, p=25, verbose=False)[0]
                h_est = centrer_le_noyau(h_est)
                if my_tv:
                    im_y_rec, _ = tv_deconv(v, h_est, add_tapping=add_tapping)
                else:
                    im_y_rec, _ = TV_deconv(v, h_est, 2000/255)
                im_ycbcr_rec = np.stack([im_y_rec, im_ycbcr[:, :, 1], im_ycbcr[:, :, 2]], axis=-1)
                im_rec = ycbcr_to_rgb(im_ycbcr_rec)
                images = [im_y, v, h, h_est, im_y_rec]
                titles = ["Original Y Channel", "Blurred Y Channel", "True Kernel", "Estimated Kernel", "Estimated Y Channel"]
                view_images(*images, titles=titles, cols=5, figsize=(15,5))
                plt.pause(1)
            else:   
                v = convolve2d(im, h, mode='same', boundary='symm')
                h_est = estime_noyau(v, p=25, verbose=False)[0]
                h_est = centrer_le_noyau(h_est)
                if my_tv:
                    im_rec, _ = tv_deconv(v, h_est, add_tapping=add_tapping)
                else:
                    im_rec, _ = TV_deconv(v, h_est, 2000/255)
                images = [im, v, h, h_est, im_rec]
                titles = ["Original Image", "Blurred Image", "True Kernel", "Estimated Kernel", "Deconvolved Image"]
                view_images(*images, titles=titles, cols=5, figsize=(15,5))
                plt.pause(1)

            metrics = compute_metrics(im, im_rec)
            print(
                    f"PSNR: {metrics['PSNR']:.2f} | "
                    f"SNR: {metrics['SNR']:.2f} | SSIM: {metrics['SSIM']:.4f}"
                )


def kernel_size_test():
    img = load_test_data()
    K = [15, 25, 35, 45]
    for u in img:
        if u.ndim == 3 and u.shape[2] == 3:
            u_ycbcr = rgb_to_ycbcr(u)
            u_y = u_ycbcr[:, :, 0]
            for k in K:
                h = motion_blur_kernel(k, 30)
                v = convolve2d(u_y, h, mode='same', boundary='symm')
                h_est = estime_noyau(v, p=k, verbose=False)[0]
                h_est = centrer_le_noyau(h_est)
                u_rec_y, _ = tv_deconv(v, h, add_tapping=add_tapping)
                u_rec_ycbcr = np.stack([u_rec_y, u_ycbcr[:, :, 1], u_ycbcr[:, :, 2]], axis=-1)
                u_rec = ycbcr_to_rgb(u_rec_ycbcr)
                images = [u_y, v, h, h_est, u_rec_y]
                titles = ["Original Y Channel", "Blurred Y Channel", f"True Kernel (size={k})", f"Estimated Kernel (size={k})", f"Deconvolved Y Channel"]
                view_images(*images, titles=titles, cols=5, figsize=(15,5))
                plt.pause(1)
        else:   
            for k in K:
                h = motion_blur_kernel(k, 30)
                v = convolve2d(u, h, mode='same', boundary='symm')
                h_est = estime_noyau(v, p=k, verbose=False)[0]
                h_est = centrer_le_noyau(h_est)
                u_rec, _ = tv_deconv(v, h_est, add_tapping=add_tapping)
                images = [u, v, h, h_est, u_rec_y]
                titles = ["Original Image", "Blurred Image", f"True Kernel (size={k})", f"Estimated Kernel (size={k})", f"Deconvolved Image"]
                view_images(*images, titles=titles, cols=5, figsize=(15,5))
                plt.pause(1)
        
        metrics = compute_metrics(u, u_rec)
        print(
            f"PSNR: {metrics['PSNR']:.2f} | "
            f"SNR: {metrics['SNR']:.2f} | SSIM: {metrics['SSIM']:.4f}"
        )
            


def compressed_test():
    img = load_test_data()[0] # img png (compressed without loss)
    jpeg_img = png_to_jpeg(img, quality=90) # jpeg compression (with loss)
    REC = []
    for u in [img, jpeg_img]:
        if u.ndim == 3 and u.shape[2] == 3:
            u_ycbcr = rgb_to_ycbcr(u)
            u_y = u_ycbcr[:, :, 0]
            v = u_y
            h_est = estime_noyau(v, p=25, verbose=False)[0]
            h_est = centrer_le_noyau(h_est)
            u_rec_y, _ = tv_deconv(v, h_est, add_tapping=add_tapping)
            u_rec_ycbcr = np.stack([u_rec_y, u_ycbcr[:, :, 1], u_ycbcr[:, :, 2]], axis=-1)
            u_rec = ycbcr_to_rgb(u_rec_ycbcr)
            REC.append(u_rec)
            images = [u_y, h_est, u_rec_y]
            titles = ["Original Y Channel"] if u is img else ["JPEG Image Y Channel"]
            titles += ["Estimated Kernel", "Deconvolved Y Channel"]
            view_images(*images, titles=titles, cols=3, figsize=(16,4))
            plt.pause(1)
        else:
            v = u
            h_est = estime_noyau(v, p=25, verbose=False)[0]
            h_est = centrer_le_noyau(h_est)
            u_rec, _ = tv_deconv(v, h_est, add_tapping=add_tapping)
            REC.append(u_rec)
            images = [u, h_est, u_rec]
            titles = ["Original Image"] if u is img else ["JPEG Image"]
            titles += ["Estimated Kernel", "Deconvolved Image"]
            view_images(*images, titles=titles, cols=3, figsize=(16,4))
            plt.pause(1)

    view_images(img, jpeg_img, REC[0], REC[1], titles=["Original PNG Image", "JPEG Compressed Image", "Deconvolved PNG Image", "Deconvolved Image from JPEG"], cols=4, figsize=(12,4))
    plt.pause(np.inf)




if __name__ == "__main__":
    # h = motion_blur_kernel(15, 30)
    # plt.imshow(h, cmap='gray')
    # plt.title("Motion Blur Kernel")     
    # plt.show()

    # kernel_estimation_test()
    # test_tv_deconv()
    # kernel_size_test()
    # noise_test()
    # kernel_size_test()
    # kernel_estimation_test()
    compressed_test()