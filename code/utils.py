import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional
from skimage import data
import random as rd
import os





class ImageHandler:

    def __init__(self):
      
        self.image = None

    def load_image(self, path: str, as_gray=True, normalize=True, random = False):
        """Load an image from the specified path.
        Args:
            as_gray: if True and the image is RGB, convert to grayscale using luma coefficients.
            normalize: if True, scale to [0,1] (only if max>0).
        Returns:
            np.ndarray float64 image.
        """
        if random:
            availables = [data.astronaut(), data.camera(), data.coins(), data.moon(), data.page(), data.rocket(), data.text()]
            img = rd.choice(availables)
        else:
            image_path = path
            try:
                img = plt.imread(image_path)
            except:
                raise FileNotFoundError(f"Image file not found: {image_path}")
        img = img.astype(np.float32, copy=False)
        if as_gray and img.ndim == 3:
            img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
        if normalize:
            vmax = np.max(img)
            if vmax > 0:
                img = img / vmax
        return img
    
    def view_images(self, *images, titles: Optional[List[str]] = None, cmap: str = 'gray', cols: int = 3, figsize: tuple = (10, 4)):
        """Display one or multiple images in a grid.
        Args:
            *images: images to display; if empty, displays self.image.
            titles: optional list of titles (same length as images).
            cmap: matplotlib colormap to use (for grayscale).
            cols: number of columns in the grid.
            figsize: figure size.
        """
        plt.close('all')
        imgs = images if len(images) > 0 else (self.image,)
        n = len(imgs)
        cols = min(cols, n)
        rows = int(np.ceil(n / cols))

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
        plt.show()

    def show(self, cmap: str = 'gray', title: Optional[str] = None):
        """Quickly show the currently loaded image."""
        plt.figure()
        if self.image.ndim == 2:
            plt.imshow(self.image, cmap=cmap)
        else:
            plt.imshow(self.image)
        if title:
            plt.title(title)
        plt.axis('off')
        plt.show()

    def save(self, out_path: str, img: Optional[np.ndarray] = None, cmap: str = 'gray'):
        """Save an image to disk. If img is None, saves the current image."""
        arr = self.image if img is None else img
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.clip(arr, 0.0, 1.0)
        plt.imsave(out_path, arr, cmap=cmap if arr.ndim == 2 else None)

    def to_gray(self, image: np.ndarray) -> np.ndarray:
        """Convert an RGB image to grayscale using luma coefficients."""
        if image.ndim == 3:
            return np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
        return image

    def to_rgb(self, image: np.ndarray) -> np.ndarray:
        """Convert a grayscale image to RGB by repeating the channels."""
        if image.ndim == 2:
            return np.stack([image] * 3, axis=-1)
        return image
    
    def RGB_to_YCbCr(self, img: np.ndarray, normalized=True) -> np.ndarray:
        """Convert an RGB image to YCbCr color space."""
        if img.shape[2] == 3:
            transform_matrix = np.array([[65.481, 128.553, 24.966],
                                         [-37.797, -74.203, 112.0],
                                         [112.0, -93.786, -18.214]])
            shift = np.array([16, 128, 128])
            if normalized:
                ycbcr = img @ transform_matrix.T + shift
                ycbcr = ycbcr / 255.0
            else:
                ycbcr = (img @ transform_matrix.T) / 255.0 + shift
            return ycbcr
        raise ValueError("Input image must be an RGB image with 3 channels.")
    
    def YCbCr_to_RGB(self, ycbcr: np.ndarray, Cr: Optional[np.ndarray] = None, Cb: Optional[np.ndarray] = None, normalized=True) -> np.ndarray:
        # Attendu: ordre [Y, Cb, Cr]
        if ycbcr.ndim == 2:
            if Cr is None or Cb is None:
                raise ValueError("Cr et Cb doivent être fournis si ycbcr est 2D (Y seul).")
            ycbcr = np.stack([ycbcr, Cb, Cr], axis=-1)

        if ycbcr.ndim != 3 or ycbcr.shape[2] != 3:
            raise ValueError("Input YCbCr doit être (H,W,3) ou Y seul + Cb,Cr.")

        if normalized:
            Y  = ycbcr[:, :, 0] * 255.0
            Cb = ycbcr[:, :, 1] * 255.0
            Cr = ycbcr[:, :, 2] * 255.0
        else:
            Y, Cb, Cr = ycbcr[:, :, 0], ycbcr[:, :, 1], ycbcr[:, :, 2]

        Y  = Y  - 16.0
        Cb = Cb - 128.0
        Cr = Cr - 128.0

        R = 1.164 * Y + 1.596 * Cr
        G = 1.164 * Y - 0.392 * Cb - 0.813 * Cr
        B = 1.164 * Y + 2.017 * Cb

        rgb8 = np.stack([R, G, B], axis=-1)

        if normalized:
            return np.clip(rgb8, 0.0, 255.0).astype(np.float32) / 255.0
        else:
            return np.clip(rgb8, 0, 255).astype(np.uint8)

if __name__ == "__main__":
    cwd = os.getcwd()
    path = ""
    handler = ImageHandler()
    img = handler.load_image(path, as_gray=True, normalize=True, random = True)
    handler.view_images(img, titles=["Loaded Image"])
