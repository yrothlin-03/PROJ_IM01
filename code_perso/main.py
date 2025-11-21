import argparse
from .tvdeconv import tv_deconv_circular
from .kernel_estimation import blur_kernel_estimation
from .kernel_estimation_bis import estime_noyau, centrer_le_noyau
from .utils import rgb_to_ycbcr, ycbcr_to_rgb
import numpy as np
import imageio

kernel_estimation = estime_noyau



def argparser():
    parser = argparse.ArgumentParser(description="Kernel estimation from a blurred image using Goldstein & Fattal's method.")
    parser.add_argument("-input_image", type=str, help="Path to the input blurred image.")
    parser.add_argument("-output_kernel", type=str, help="Path to save the estimated blur kernel.")
    parser.add_argument("-output_deconvolved", type=str, help="Path to save the deconvolved image.")
    return parser


def main():
    parser = argparser()
    args = parser.parse_args()

    img = imageio.imread(args.input_image).astype(np.float32) / 255.0
    if img.ndim == 3:
        img_yrcbcr = rgb_to_ycbcr(img)
        v = img_yrcbcr[..., 0]
    else:
        v = img
    
    h_est = kernel_estimation(v, p=25, verbose=True)
    h_est = centrer_le_noyau(h_est)
    u_rec = tv_deconv_circular(v, h_est, add_tapping=True)

    if img.ndim == 3:
        img_ycbcr_rec = img_yrcbcr.copy()
        img_ycbcr_rec[..., 0] = u_rec
        img_rec = ycbcr_to_rgb(img_ycbcr_rec)
    else:
        img_rec = u_rec
    
    imageio.imwrite(args.output_kernel, (h_est / h_est.max() * 255).astype(np.uint8))
    print(f"Estimated kernel saved to {args.output_kernel}")
    imageio.imwrite(args.output_deconvolved, (img_rec * 255).astype(np.uint8))
    print(f"Deconvolved image saved to {args.output_deconvolved}")

if __name__ == "__main__":
    main()