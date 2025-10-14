import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional





class ImageHandler:

    def __init__(self, image_path):
        self.image_path = image_path
        self.image = self.load_image()
    

    def load_image(self, as_gray=True, normalize=True):
        """Load an image from the specified path.
        Args:
            as_gray: if True and the image is RGB, convert to grayscale using luma coefficients.
            normalize: if True, scale to [0,1] (only if max>0).
        Returns:
            np.ndarray float64 image.
        """
        try:
            img = plt.imread(self.image_path)
        except:
            raise FileNotFoundError(f"Image file not found: {self.image_path}")
        img = img.astype(np.float64, copy=False)
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
        imgs = images if len(images) > 0 else (self.image,)
        n = len(imgs)
        cols = max(1, int(cols))
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
