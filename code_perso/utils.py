import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional
from skimage import data, io, img_as_ubyte
from skimage.transform import resize
from skimage.metrics import structural_similarity as ssim
from pathlib import Path
import os
import imageio.v3 as iio




def load_test_data(mode: str = '0000000001'):
    dir_path = Path(__file__).resolve().parent.parent / "data"
    images_paths = [
        dir_path / "taj_mahal.png",
        dir_path / "malll.png",
        dir_path / "rochers_mer.png",
        dir_path / "arbres.png",
        dir_path / "perso1.png",
        dir_path / "perso2.png",
        dir_path / "perso3.png",
        dir_path / "perso4.png",
        dir_path / "lena.png",
    ]
    images = []
    if mode[0]=='1':
        images.append(data.astronaut().astype(np.float32) / 255.0)
    for path in images_paths:
        if path.exists():
            if mode[images_paths.index(path)+1]=='1':
                img = io.imread(path).astype(np.float32) / 255.0
                images.append(img)
        else:
            raise FileNotFoundError(f"Warning: {path} does not exist.")
    return images



    
def view_images(*images, titles: Optional[List[str]] = None, title: Optional[str]=None, cmap: str = 'gray', cols: int = 3, block: bool = False):
    """Display one or multiple images in a grid."""
    plt.close('all')
    if len(images) > 0:
        imgs = images
    else:
        raise ValueError("No images to display.")
    n = len(imgs)
    cols = min(cols, n)
    rows = int(np.ceil(n / cols))

    w_per_img = 4
    h_per_img = 3
    figsize = (w_per_img * cols, h_per_img * rows)
    fig = plt.figure(figsize=figsize)
    for idx, im in enumerate(imgs, start=1):
        ax = fig.add_subplot(rows, cols, idx)
        if im.ndim == 2:
            ax.imshow(im, cmap=cmap)
        else:
            ax.imshow(im)
        if titles and idx-1 < len(titles):
            ax.set_title(titles[idx-1])
        ax.axis('off')
    plt.tight_layout()

    plt.savefig(f"results/{title}.png", dpi=300, bbox_inches="tight")
    plt.show(block=block)




def rgb_to_ycbcr(img: np.ndarray) -> np.ndarray:
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("L'image doit être RGB avec shape (H, W, 3).")

    if np.max(img) > 1.0:
        img = img.astype(np.float32) / 255.0 

    R = img[..., 0]
    G = img[..., 1]
    B = img[..., 2]

    Y  = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.168736 * R - 0.331264 * G + 0.5 * B + 0.5
    Cr = 0.5 * R - 0.418688 * G - 0.081312 * B + 0.5

    ycbcr = np.stack((Y, Cb, Cr), axis=-1)
    return np.clip(ycbcr, 0.0, 1.0).astype(np.float32)




def ycbcr_to_rgb(ycbcr: np.ndarray) -> np.ndarray:
    if ycbcr.ndim != 3 or ycbcr.shape[2] != 3:
        raise ValueError("L'image doit être YCbCr avec shape (H, W, 3).")

    Y  = ycbcr[..., 0]
    Cb = ycbcr[..., 1] - 0.5
    Cr = ycbcr[..., 2] - 0.5

    R = Y + 1.402 * Cr
    G = Y - 0.344136 * Cb - 0.714136 * Cr
    B = Y + 1.772 * Cb

    rgb = np.stack((R, G, B), axis=-1)
    return np.clip(rgb, 0.0, 1.0).astype(np.float32)
    


def add_noise(image, mean: float=0, std: float = 1):
    noise = np.random.normal(mean, std, size=image.shape)
    noisy_image = image + noise
    return np.clip(noisy_image, 0.0, 1.0) if image.max() <= 1.0 else np.clip(noisy_image, 0, 255)
    


def motion_blur_kernel(size: int, angle: float) -> np.ndarray:
    """Generate a motion blur kernel of given size and angle."""
    kernel = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    angle = np.deg2rad(angle)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)

    for i in range(size):
        x = int(center + (i - center) * cos_a)
        y = int(center + (i - center) * sin_a)
        if 0 <= x < size and 0 <= y < size:
            kernel[y, x] = 1

    kernel /= np.sum(kernel)
    return kernel



def downsampling(image, k: int = 2):
    h, w = image.shape[:2]
    return resize(image, (h // k, w // k), anti_aliasing=True)



    
def PSNR(original: np.ndarray, reconstructed: np.ndarray, max_pixel: float = 1.0) -> float:
    """Compute the Peak Signal-to-Noise Ratio (PSNR) between two images."""
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(max_pixel / np.sqrt(mse))

def SNR(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Compute the Signal-to-Noise Ratio (SNR) between two images."""
    signal_power = np.mean(original ** 2)
    noise_power = np.mean((original - reconstructed) ** 2)
    if noise_power == 0:
        return float('inf')
    return 10 * np.log10(signal_power / noise_power)

def SSIM(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Compute the Structural Similarity Index (SSIM) between two images."""
    if original.shape != reconstructed.shape:
        raise ValueError("Original and reconstructed images must have the same shape for SSIM.")

    original = original.astype(np.float32)
    reconstructed = reconstructed.astype(np.float32)

    h, w = original.shape[:2]
    min_side = min(h, w)
    if min_side < 3:
        raise ValueError("Images are too small for SSIM (minimum side length is 3 pixels).")
    win_size = 7 if min_side >= 7 else (min_side if min_side % 2 == 1 else min_side - 1)

    if original.ndim == 3 and original.shape[2] == 3:
        ssim_value = ssim(original, reconstructed, channel_axis=-1, win_size=win_size, data_range=1.0)
    else:
        ssim_value = ssim(original, reconstructed, win_size=win_size, data_range=1.0)

    return ssim_value
    

def compute_metrics(u: np.ndarray, u_rec: np.ndarray) -> dict:
    """Compute PSNR, SNR, and SSIM between two images."""
    metrics = {
        'PSNR': PSNR(u, u_rec),
        'SNR': SNR(u, u_rec),
        'SSIM': SSIM(u, u_rec)
    }
    return metrics


def png_to_jpeg(img: np.ndarray, quality: int = 90) -> np.ndarray:
    """Convert a PNG image to JPEG format with specified quality."""
    img_uint8 = img_as_ubyte(img)
    iio.imwrite("temp.jpg", img_uint8, quality=quality)
    jpeg_img = io.imread("temp.jpg").astype(np.float32) / 255.0
    os.remove("temp.jpg")
    return jpeg_img





def gaussian_kernel(n, s):
    a = np.arange(-(n//2), n//2+1)
    X, Y = np.meshgrid(a, a)
    k = np.exp(-(X*X + Y*Y)/(2*s*s))
    return k / k.sum()




if __name__ == "__main__":
    pass
