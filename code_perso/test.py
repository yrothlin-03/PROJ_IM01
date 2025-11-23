from kernel_estimation_bis import estime_noyau, centrer_le_noyau
from kernel_estimation import blur_kernel_estimation
# from kernel_estimation import center_kernel
from tv_deconv import tv_deconv as tv_deconv_symm
from tvdeconv import tv_deconv_circular
from utils import *
import numpy as np
from scipy.signal import convolve2d
from time import time
import matplotlib.pyplot as plt


plt.ion()   


TV_deconv = tv_deconv_circular
# TV_deconv = tv_deconv_symm



add_tapping = True 

apply_motion_blur = True

kernel_estimation = estime_noyau
# kernel_estimation = blur_kernel_estimation



" ------------------------- TV Deconvolution Test ------------------   "
" ------------------------- TV Deconvolution Test ------------------   "
" ------------------------- TV Deconvolution Test ------------------   "
" ------------------------- TV Deconvolution Test ------------------   "
" ------------------------- TV Deconvolution Test ------------------   "


def test_circular_vs_symmetric():
    img = load_test_data()[0]
    if img.ndim == 3 and img.shape[2] == 3:
        img_ycbcr = rgb_to_ycbcr(img)
        img_y = img_ycbcr[:, :, 0]
        u = img_y
    else:
        u = img
    h = motion_blur_kernel(15, 60)
    v = convolve2d(u, h, mode='same', boundary='symm')
    u_rec_symm = tv_deconv_symm(v, h, add_tapping=add_tapping)
    u_rec_circ = tv_deconv_circular(v, h, add_tapping=add_tapping)
    metrics_symm = compute_metrics(u, u_rec_symm)
    metrics_circ = compute_metrics(u, u_rec_circ)
    images = [u, v, h, u_rec_symm, u_rec_circ]
    titles = ["Original Image", "Blurred Image", "Blur Kernel", f"Deconv Symmetric (PSNR: {metrics_symm['PSNR']:.2f})", f"Deconv Circular (PSNR: {metrics_circ['PSNR']:.2f})"]
    view_images(*images, title="circular_vs_symmetric",titles=titles, cols=5, figsize=(15,5))
    plt.pause(5)
    plt.close()
    h_est = estime_noyau(v, p=15, verbose=False)
    h_est = centrer_le_noyau(h_est)
    u_rec_symm_est = tv_deconv_symm(v, h_est, lam=2000, add_tapping=add_tapping)
    u_rec_circ_est = tv_deconv_circular(v, h_est, lam=2000, add_tapping=add_tapping)
    metrics_symm_est = compute_metrics(u, u_rec_symm_est)
    metrics_circ_est = compute_metrics(u, u_rec_circ_est)
    images = [u, v, h_est, u_rec_symm_est, u_rec_circ_est]
    titles = ["Original Image", "Blurred Image", "Estimated Kernel", f"Deconv Symm Est (PSNR: {metrics_symm_est['PSNR']:.2f})", f"Deconv Circ Est (PSNR: {metrics_circ_est['PSNR']:.2f})"]
    view_images(*images, title="circular_vs_symmetric_estimated",titles=titles, cols=5)
    plt.pause(5)
    plt.close()

def test_tv_deconv():
    images = load_test_data()
    for u in images:
        if u.ndim == 3 and u.shape[2] == 3:                             # RGB image
            u_ycbcr = rgb_to_ycbcr(u)
            u_y = u_ycbcr[:, :, 0]
            h = motion_blur_kernel(25, 60)
            v = convolve2d(u_y, h, mode='same', boundary='symm')
            h_rec = kernel_estimation(v, p=25, verbose=False)
            print("Shape of estimated kernel:", h_rec.shape)
            h_rec = centrer_le_noyau(h_rec)
            t1 = time()
            u_rec = TV_deconv(v, h, lam = 2000, add_tapping=add_tapping)
            u_rec2 = TV_deconv(v, h_rec, lam = 2000, add_tapping=add_tapping)
            t2 = time()
            dt = t2 - t1
            print(f"TV Deconvolution took {dt:.2f} seconds.")

            u_rec_ycbcr = np.stack([u_rec, u_ycbcr[:, :, 1], u_ycbcr[:, :, 2]], axis=-1)
            u_rec_rgb = ycbcr_to_rgb(u_rec_ycbcr)

            urec2_ycbcr = np.stack([u_rec2, u_ycbcr[:, :, 1], u_ycbcr[:, :, 2]], axis=-1)
            u_rec2_rgb = ycbcr_to_rgb(urec2_ycbcr)

            u_rec = u_rec_rgb
            u_rec2 = u_rec2_rgb

            metrics = compute_metrics(u, u_rec)
            print(f"PSNR: {metrics['PSNR']:.2f} | SNR: {metrics['SNR']:.2f} | SSIM: {metrics['SSIM']:.4f}")
            metrics2 = compute_metrics(u, u_rec2)
            print(f"PSNR (estimated kernel): {metrics2['PSNR']:.2f} | SNR: {metrics2['SNR']:.2f} | SSIM: {metrics2['SSIM']:.4f}")

        else:                                                           # Grayscale image       
            h = motion_blur_kernel(25, 60)
            v = convolve2d(u, h, mode='same', boundary='symm')
            h_rec = estime_noyau(v, p=25, verbose=False)
            h_rec = centrer_le_noyau(h_rec)
            t1 = time()
            u_rec = TV_deconv(v, h, lam = 2000, add_tapping=add_tapping)
            u_rec2 = TV_deconv(v, h_rec, lam = 2000, add_tapping=add_tapping)
            t2 = time()
            dt = t2 - t1
            print(f"TV Deconvolution took {dt:.2f} seconds.")

            metrics = compute_metrics(u, u_rec)
            print(f"PSNR: {metrics['PSNR']:.2f} | SNR: {metrics['SNR']:.2f} | SSIM: {metrics['SSIM']:.4f}")
            metrics2 = compute_metrics(u, u_rec2)
            print(f"PSNR (estimated kernel): {metrics2['PSNR']:.2f} | SNR: {metrics2['SNR']:.2f} | SSIM: {metrics2['SSIM']:.4f}")

        images = [u, v, h, h_rec, u_rec, u_rec2]
        titles = ["Original Image", "Blurred Image", "Blur Kernel", "Estimated Kernel", f"Deconv IMG \n (PSNR: {metrics['PSNR']:.2f})", f"Deconv IMG Estimated Kernel \n (PSNR: {metrics2['PSNR']:.2f})"]
        view_images(*images, title="true_vs_blur_deconv",titles=titles, cols=3)
        plt.pause(np.inf)




def test_hyperparameters_tv_deconv():

    LAMB = [100, 1000, 2000, 10000]
    GAM = [0.1, 1, 5, 20]

    img = load_test_data(mode='1000000000')[0]
    if img.ndim == 3 and img.shape[2] == 3:
        img_ycbcr = rgb_to_ycbcr(img)
        img_y = img_ycbcr[:, :, 0]
        u = img_y
    else:
        u = img

    h = motion_blur_kernel(15, 60)
    v = convolve2d(u, h, mode='same', boundary='symm')
    UREC = []
    images = []
    titles = []
    for lamb in LAMB:
        for gam in GAM:
            u_rec = TV_deconv(v, h, lam=lamb, gamma=gam, add_tapping=add_tapping)
            if img.ndim == 3 and img.shape[2] == 3:
                u_rec_ycbcr = np.stack([u_rec, img_ycbcr[:, :, 1], img_ycbcr[:, :, 2]], axis=-1)
                u_rec_rgb = ycbcr_to_rgb(u_rec_ycbcr)
                u_rec = u_rec_rgb
                metrics = compute_metrics(img, u_rec)
            UREC.append((gam, u_rec))
        images = images + [img] + [u_rec for _, u_rec in UREC[-len(GAM):]]
        titles = titles + ["Original Image"] + [f"Deconv (λ={lamb}, γ={gam})\n(PSNR: {metrics['PSNR']:.2f})" for gam, _ in UREC[-len(GAM):]]
    view_images(*images, titles=titles, title= "hyperparam_test_circ", cols=len(GAM)+1)
    plt.pause(10)


def hyperparameter_tuning_graph():
    img = load_test_data(mode='1000000000')[0]
    if img.ndim == 3 and img.shape[2] == 3:
        img_ycbcr = rgb_to_ycbcr(img)
        img_y = img_ycbcr[:, :, 0]
        u = img_y
    else:
        u = img
    h = motion_blur_kernel(25, 60)
    v = convolve2d(u, h, mode='same', boundary='symm')
    L = np.linspace(100, 20000, 30)
    G = np.linspace(0.1, 50, 20)
    PSNR_VALUES = np.zeros((len(L), len(G)))
    for i, lamb in enumerate(L):
        for j, gam in enumerate(G):
            u_rec = TV_deconv(v, h, lam=lamb, gamma=gam, add_tapping=add_tapping)
            if img.ndim == 3 and img.shape[2] == 3:
                u_rec_ycbcr = np.stack([u_rec, img_ycbcr[:, :, 1], img_ycbcr[:, :, 2]], axis=-1)
                u_rec_rgb = ycbcr_to_rgb(u_rec_ycbcr)
                u_rec = u_rec_rgb
            metrics = compute_metrics(img, u_rec)
            PSNR_VALUES[i, j] = metrics['PSNR']
    max_idx = np.unravel_index(np.argmax(PSNR_VALUES), PSNR_VALUES.shape)
    lambda_max = L[max_idx[0]]
    gamma_max = G[max_idx[1]]
    max_psnr = PSNR_VALUES[max_idx]
    G_mesh, L_mesh = np.meshgrid(G, L)
    plt.figure(figsize=(10, 6))
    cp = plt.contourf(G_mesh, L_mesh, PSNR_VALUES, levels=20, cmap='viridis')
    plt.colorbar(cp)
    plt.xlabel('Gamma (γ)')
    plt.ylabel('Lambda (λ)')
    plt.title('PSNR values for different hyperparameters')
    plt.xticks(G[::3])   
    plt.yticks(L[::3])
    plt.grid(True, linestyle=':', linewidth=0.5)
    plt.axhline(lambda_max, linestyle='--')
    plt.axvline(gamma_max, linestyle='--')
    plt.scatter(gamma_max, lambda_max, marker='o')
    plt.text(
        gamma_max,
        lambda_max,
        f"  Max PSNR={max_psnr:.2f}\n(λ={lambda_max:.0f}, γ={gamma_max:.2f})",
        fontsize=8,
        verticalalignment='bottom',
    )
    plt.savefig(f"./results/hyperparameter_tuning.png", dpi=300, bbox_inches="tight")
    plt.show()

    


def test_tvdeconv_noise():
    img = load_test_data()[0]
    noise_levels = [0, 1, 5, 10, 20]
    images, images_noisy = [], []
    titles, titles_noisy = [], []

    if img.ndim == 3 and img.shape[2] == 3:
        img_ycbcr = rgb_to_ycbcr(img)
        img_y = img_ycbcr[:, :, 0]
        u = img_y
    else:
        u = img
    h = motion_blur_kernel(15, 60)
    for std in noise_levels:
        noise = np.random.normal(0, std/255, u.shape) 
        v = convolve2d(u, h, mode='same', boundary='symm')
        v = v + noise
        u_rec = TV_deconv(v, h, add_tapping=add_tapping)
        if img.ndim == 3 and img.shape[2] == 3:
            v_rgb = np.stack([v, img_ycbcr[:, :, 1], img_ycbcr[:, :, 2]], axis=-1)
            v_rgb = ycbcr_to_rgb(v_rgb)
            u_rec_ycbcr = np.stack([u_rec, img_ycbcr[:, :, 1], img_ycbcr[:, :, 2]], axis=-1)
            u_rec_rgb = ycbcr_to_rgb(u_rec_ycbcr)
            u_rec = u_rec_rgb
        metrics = compute_metrics(u, u_rec)
        images_noisy.append(v if img.ndim == 2 else v_rgb)
        titles_noisy.append(f"Noisy Blurred Image (std={std})")
        images.append(u_rec)
        titles.append(f"Deconv Noisy (std={std})\n(PSNR: {metrics['PSNR']:.2f})")
    
    images = images + images_noisy
    titles = titles + titles_noisy
    view_images(*images, title="tvdeconv_noise_test",titles=titles, cols=len(images)//2)
    plt.pause(10)
    plt.close()







" ------------------------- Kernel estimation Test ------------------ "
" ------------------------- Kernel estimation Test ------------------ "
" ------------------------- Kernel estimation Test ------------------ "
" ------------------------- Kernel estimation Test ------------------ "
" ------------------------- Kernel estimation Test ------------------ "


def noise_test():
    img = load_test_data()[0]
    noise_levels = [0, 1, 5, 10, 20]
    u = img
    images = []
    titles = []
    for std in noise_levels:
        if u.ndim == 3 and u.shape[2] == 3:
            u_ycbcr = rgb_to_ycbcr(u)
            u_y = u_ycbcr[:, :, 0]
            noise = np.random.normal(0, std/255, u_y.shape)
            h = motion_blur_kernel(15, 60)
            v = convolve2d(u_y, h, mode='same', boundary='symm')
            v = v + noise
            h_est = kernel_estimation(v, p=15, verbose=False)
            h_est = centrer_le_noyau(h_est)
            u_rec_y = TV_deconv(v, h_est, add_tapping=add_tapping)
            u_rec_ycbcr = np.stack([u_rec_y, u_ycbcr[:, :, 1], u_ycbcr[:, :, 2]], axis=-1)
            u_rec = ycbcr_to_rgb(u_rec_ycbcr)
            metrics = compute_metrics(u, u_rec)
        else:
            noise = np.random.normal(0, std/255, u.shape)
            v = u
            h = motion_blur_kernel(15, 60)
            v = convolve2d(v, h, mode='same', boundary='symm')
            v = v + noise
            h_est = kernel_estimation(v, p=15, verbose=False)
            h_est = centrer_le_noyau(h_est)
            u_rec = TV_deconv(v, h_est, add_tapping=add_tapping)
            metrics = compute_metrics(u, u_rec)

        images = images + [u, v, h, h_est, u_rec]
        titles = titles + ["Original Image", f"Noisy Image (std={std})", "True Kernel", "Estimated Kernel", f"Deconvolved Image (PSNR: {metrics['PSNR']:.2f})"]
    view_images(*images, title = f"noise_test", titles=titles, cols=5)
    plt.pause(1)



def realimages_test():
    Images = load_test_data('0101101010')
    images = []
    titles = []
    for i, u in enumerate(Images):
        if u.ndim == 3 and u.shape[2] == 3:
            u_ycbcr = rgb_to_ycbcr(u)
            u_y = u_ycbcr[:, :, 0]
            v = u_y
            h_est = kernel_estimation(v, p=25, verbose=False)
            h_est = centrer_le_noyau(h_est)
            u_rec_y = TV_deconv(v, h_est, add_tapping=add_tapping)
            u_rec_ycbcr = np.stack([u_rec_y, u_ycbcr[:, :, 1], u_ycbcr[:, :, 2]], axis=-1)
            u_rec = ycbcr_to_rgb(u_rec_ycbcr)
            v = ycbcr_to_rgb(np.stack([v, u_ycbcr[:, :, 1], u_ycbcr[:, :, 2]], axis=-1))
        else:
            v = u
            h_est = kernel_estimation(v, p=25, verbose=False)
            h_est = centrer_le_noyau(h_est)
            u_rec = TV_deconv(v, h_est, add_tapping=add_tapping)

        images = images + [u, h_est, u_rec]
        titles = titles + ["Original Image", "Estimated Kernel", "Deconvolved Image"]
    view_images(*images, title=f"real_images_test",titles=titles, cols=3)
    plt.pause(1)   



# def kernel_estimation_test():
#     img = load_test_data()
#     K = [motion_blur_kernel(25, k) for k in [0, 45, 90]]
#     K.append(gaussian_kernel(25, 5))
#     for h in K:
#         for im in img:
#             if im.ndim == 3 and im.shape[2] == 3:
#                 im_ycbcr = rgb_to_ycbcr(im)
#                 im_y = im_ycbcr[:, :, 0]
#                 v = convolve2d(im_y, h, mode='same', boundary='symm')
#                 h_est = kernel_estimation(v, p=25, verbose=True)
#                 plt.close()
#                 plt.imshow(h_est, cmap='gray')
#                 plt.title("Estimated Kernel")
#                 plt.show()
#                 plt.pause(5)
#                 plt.close()
#                 h_est = center_kernel(h_est)
#                 im_y_rec = TV_deconv(v, h_est, add_tapping=add_tapping)
#                 im_ycbcr_rec = np.stack([im_y_rec, im_ycbcr[:, :, 1], im_ycbcr[:, :, 2]], axis=-1)
#                 im_rec = ycbcr_to_rgb(im_ycbcr_rec)
#                 images = [im_y, v, h, h_est, im_y_rec]
#                 titles = ["Original Y Channel", "Blurred Y Channel", "True Kernel", "Estimated Kernel", "Estimated Y Channel"]
#                 view_images(*images, titles=titles, cols=5, figsize=(15,5))
#                 plt.pause(1)
#             else:   
#                 v = convolve2d(im, h, mode='same', boundary='symm')
#                 h_est = kernel_estimation(v, p=25, verbose=False)
#                 h_est = center_kernel(h_est)
#                 im_rec = TV_deconv(v, h_est, add_tapping=add_tapping)
#                 images = [im, v, h, h_est, im_rec]
#                 titles = ["Original Image", "Blurred Image", "True Kernel", "Estimated Kernel", "Deconvolved Image"]
#                 view_images(*images, titles=titles, cols=5, figsize=(15,5))
#                 plt.pause(1)

#             metrics = compute_metrics(im, im_rec)
#             print(
#                     f"PSNR: {metrics['PSNR']:.2f} | "
#                     f"SNR: {metrics['SNR']:.2f} | SSIM: {metrics['SSIM']:.4f}"
#                 )


def kernel_size_test():
    img = load_test_data()[0]
    K = [15, 25, 35]
    images = []
    titles = []
    for k in K:
        if img.ndim == 3 and img.shape[2] == 3:
            img_ycbcr = rgb_to_ycbcr(img)
            img_y = img_ycbcr[:, :, 0]
            u = img_y
        else:
            u = img
        h = motion_blur_kernel(k, 60)
        v = convolve2d(u, h, mode='same', boundary='symm')
        h_est = kernel_estimation(v, p=k, verbose=False)
        h_est = centrer_le_noyau(h_est)
        u_rec = TV_deconv(v, h_est, add_tapping=add_tapping)
        if img.ndim == 3 and img.shape[2] == 3:
            u_rec_ycbcr = np.stack([u_rec, img_ycbcr[:, :, 1], img_ycbcr[:, :, 2]], axis=-1)
            u_rec_rgb = ycbcr_to_rgb(u_rec_ycbcr)
            u_rec = u_rec_rgb
        metrics = compute_metrics(img, u_rec)
        images = images + [u, v, h_est, u_rec]
        titles = titles + ["Original Image", f"Blurred Image (K={k})", "Estimated Kernel", f"Deconvolved Image (PSNR: {metrics['PSNR']:.2f})"]
    view_images(*images, title="kernel_size_test",titles=titles, cols=4)
    plt.pause(1)


def compressed_test():
    img = load_test_data('0100000000')[0] # img png (compressed without loss)
    images = []
    titles = []

    jpeg_img75 = png_to_jpeg(img, quality=75) # jpeg compression (with loss : 75% quality)
    jpeg_img50 = png_to_jpeg(img, quality=50) # jpeg compression (with loss : 50% quality)
    jpeg_img25 = png_to_jpeg(img, quality=25) # jpeg compression (with loss : 25% quality)
    jpeg_img10 = png_to_jpeg(img, quality=10) # jpeg compression (with loss : 10% quality)

    if img.ndim == 3 and img.shape[2] == 3:
        img_ycbcr = rgb_to_ycbcr(img)
        img_y = img_ycbcr[:, :, 0]
        v = img_y
    else:
        v = img
    h_est = kernel_estimation(v , p=25, verbose=False)
    h_est = centrer_le_noyau(h_est)
    u_rec = TV_deconv(v, h_est, add_tapping=add_tapping)
    if img.ndim == 3 and img.shape[2] == 3:
        u_rec_ycbcr = np.stack([u_rec, img_ycbcr[:, :, 1], img_ycbcr[:, :, 2]], axis=-1)
        u_rec_rgb = ycbcr_to_rgb(u_rec_ycbcr)
        u_rec = u_rec_rgb
    metrics = compute_metrics(img, u_rec)
    images = [img, h_est, u_rec]
    titles = ["Original Image PNG", "Estimated Kernel", f"Deconvolved Image (PSNR: {metrics['PSNR']:.2f})"]

    
    JPG = [jpeg_img75, jpeg_img50, jpeg_img25, jpeg_img10]
    Q = [75, 50, 25, 10]
    for jpeg_img, q in zip(JPG, Q):
        if img.ndim == 3 and img.shape[2] == 3:
            jpeg_img_ycbcr = rgb_to_ycbcr(jpeg_img)
            jpeg_img_y = jpeg_img_ycbcr[:, :, 0]
            v_jpeg = jpeg_img_y
        else:
            v_jpeg = jpeg_img
        h_est_jpeg = kernel_estimation(v_jpeg , p=25, verbose=False)
        h_est_jpeg = centrer_le_noyau(h_est_jpeg)
        u_rec_jpeg = TV_deconv(v_jpeg, h_est_jpeg, add_tapping=add_tapping)
        if img.ndim == 3 and img.shape[2] == 3:
            u_rec_jpeg_ycbcr = np.stack([u_rec_jpeg, jpeg_img_ycbcr[:, :, 1], jpeg_img_ycbcr[:, :, 2]], axis=-1)
            u_rec_jpeg_rgb = ycbcr_to_rgb(u_rec_jpeg_ycbcr)
            u_rec_jpeg = u_rec_jpeg_rgb
        metrics_jpeg = compute_metrics(img, u_rec_jpeg)
        images = images + [jpeg_img, h_est_jpeg, u_rec_jpeg]
        titles = titles + [f"Compressed Image JPEG (quality={q})", "Estimated Kernel", f"Deconvolved Compressed Image (PSNR: {metrics_jpeg['PSNR']:.2f})"]

    view_images(*images, title="compressed_test",titles=titles, cols=3)
    plt.pause(1)


def simple_test():
    img = load_test_data(mode='0001000000')[0]
    if img.ndim == 3 and img.shape[2] == 3:
        img_ycbcr = rgb_to_ycbcr(img)
        img_y = img_ycbcr[:, :, 0]
        v = img_y
    else:
        v = img

    h_est = kernel_estimation(v, p=25, verbose=True)
    h_est = centrer_le_noyau(h_est)
    u_rec = TV_deconv(v, h_est, add_tapping=add_tapping)
    if img.ndim == 3 and img.shape[2] == 3:
        u_rec_ycbcr = np.stack([u_rec, img_ycbcr[:, :, 1], img_ycbcr[:, :, 2]], axis=-1)
        u_rec_rgb = ycbcr_to_rgb(u_rec_ycbcr)
        u_rec = u_rec_rgb
    # metrics = compute_metrics(img, u_rec)
    # images = [img, v, h_est, u_rec]
    # titles = ["Original Image", "Blurred Image", "Estimated Kernel", f"Deconvolved Image (PSNR: {metrics['PSNR']:.2f})"]
    # view_images(*images, title="simple_test",titles=titles, cols=4)
    # plt.pause(10)

if __name__ == "__main__":

    plt.close('all')

    """ DECONVOLUTION TESTS """
    # test_circular_vs_symmetric()
    # test_hyperparameters_tv_deconv()
    # hyperparameter_tuning_graph()
    # test_tvdeconv_noise()


    """ KERNEL ESTIMATION TESTS """
    # noise_test()
    # kernel_size_test()
    # realimages_test()
    # compressed_test() 

    """ Simple test to visualize kernel estimation process """
    simple_test()
