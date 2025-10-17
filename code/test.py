from kernel_estimation_bis import estime_noyau, centrer_le_noyau
from tv_deconv import tv_deconv
from utils import ImageHandler
from pathlib import Path
import numpy as np

def snr(u, u_rec):
    """Compute the Signal-to-Noise Ratio between two images."""
    signal_power = np.sum(u ** 2)
    noise_power = np.sum((u - u_rec) ** 2)
    return 10 * np.log10(signal_power / noise_power)


def test_tv_deconv():
    pass

def test_kernel_estimation(path_to_image=None):

    handler = ImageHandler()
    if path_to_image is not None:
        img = handler.load_image(path_to_image, as_gray=False, normalize=True)
    else:
        current_file_path = Path(__file__).resolve()
        current_dir = current_file_path.parent.parent
        path_to_images = current_dir / "data" / "taj_mahal.png"
        img = handler.load_image(str(path_to_images), as_gray=False, normalize=True)
    print(f"image shape : {img.shape[2]} and type : {img.dtype}")
    if img.shape[2] != 1:
        ycbcr = handler.RGB_to_YCbCr(img)
        Y, Cb, Cr = ycbcr[:, :, 0], ycbcr[:, :, 1], ycbcr[:, :, 2]
        Y = Y.astype(np.float32)
        print("--------- Starting Kernel estimation ---------")
        h, _ = estime_noyau(Y, p=25, Nouter=3, Ntries=30, Ninner=300, verbose=False)
        h = centrer_le_noyau(h)
        print("--------- Starting deconvolution ---------")
        u, c = tv_deconv(Y, h)
        u = np.clip(u, 0.0, 1.0)
        ycbcr_rec = np.stack([u, Cb, Cr], axis=-1) 
        u_rgb = handler.YCbCr_to_RGB(ycbcr_rec, normalized=True)

        handler.view_images(img, h, u_rgb, titles=["Input Image", "Estimated Kernel", f"Deconvolved Image : SNR = {snr(img, u_rgb)}"], cols=3)
    else:
        print("--------- Starting Kernel estimation ---------")
        h, _ = estime_noyau(img, p=25, Nouter=3, Ntries=30, Ninner=300, verbose=False)
        h = centrer_le_noyau(h)
        print("--------- Starting deconvolution ---------")
        u1, c = tv_deconv(img, h)
        handler.view_images(img, h, u, titles=["Input Image", "Estimated Kernel", f"Deconvolved Image : SNR = {snr(img, u1)}"], cols=3)



if __name__ == "__main__":
    test_kernel_estimation()