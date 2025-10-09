#%% Imports
import numpy as np
from scipy.signal import convolve2d
from scipy.fft import fft2, ifft2
import scipy
from skimage import io
import time
import matplotlib.pyplot as plt
from pathlib import Path


fft = scipy.fft.rfft2
ifft = scipy.fft.irfft2
norm = np.linalg.norm

real = np.real
sin = np.sin
conj = np.conj
pi = np.pi

#%% Test for loading data
path = Path.cwd()

def load_image(name, as_gray=True):
    """Load an image from the data folder."""
    img = io.imread(path / 'images' / name, as_gray=as_gray)
    return img.astype(np.float32) / 255.0

def view_image(*images, titles=None, dpi=300, cmap='gray'):
    """
    Display one or more images side by side.
    Parameters:
        *images : one or several images (arrays)
        titles : optional list of titles for each image
        dpi : resolution of the figure
        cmap : colormap (default: 'gray')
    """
    n = len(images)
    plt.figure(figsize=(4 * n, 4), dpi=dpi)
    for i, im in enumerate(images):
        plt.subplot(1, n, i + 1)
        plt.imshow(im, cmap=cmap)
        if titles is not None and i < len(titles):
            plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()



# -------------------------------------------------------- Perso --------------------------------------------------------
# -------------------------------------------------------- Perso --------------------------------------------------------
# -------------------------------------------------------- Perso --------------------------------------------------------
# -------------------------------------------------------- Perso --------------------------------------------------------
# -------------------------------------------------------- Perso --------------------------------------------------------
# -------------------------------------------------------- Perso --------------------------------------------------------


# %% TV deconvolution 
def pad_image(im,pad=10):
    out=np.zeros((im.shape[0]+2*pad,im.shape[1]+2*pad))
    out[pad:-pad,pad:-pad]=im
    for k in range(pad):
        out[k,pad:-pad]=im[0,:]
        out[-k-1,pad:-pad]=im[-1,:]
        out[pad:-pad,k]=im[:,0]
        out[pad:-pad,-k-1]=im[:,-1]
    out[:pad,:pad]=im[0,0]
    out[-pad:,:pad]=im[-1,0]
    out[:pad,-pad:]=im[0,-1]
    out[-pad:,-pad:]=im[-1,-1]
    return out

def unpad_image(im,pad=10):
    return im[pad:-pad,pad:-pad].copy()

def taper_image(I,K):
    """ Floute une image I par le noyau K (circulairement) cela donne une image J
    On mélange l'image I avec l'image J de manière à ce que J soit prépondérente aux bords.
    L'image J, lorsqu'on la déconle par le noyau K n'aura pas d'effets de bord. """
    kh,kw=K.shape
    Ih,Iw=I.shape
    wx=np.ones((Ih,Iw),dtype=np.float32)
    wy=np.ones((Ih,Iw),dtype=np.float32)
    X,Y=np.meshgrid(np.arange(0,Iw),np.arange(0,Ih))
    wy[:kh,:]=sin(Y[:kh,:]*pi/(2*kh-1))**2
    wy[-kh:,:]=sin((Ih-Y[-kh:,:])*pi/(2*kh-1))**2
    wx[:,:kw]=sin(X[:,:kw]*pi/(2*kh-1))**2
    wx[:,-kw:]=sin((Iw-X[:,-kw:])*pi/(2*kh-1))**2
    fK=Fourier_kernel(K,I.shape)
    J=real(ifft2(fft2(I)*fK))
    out=J*(1-wx*wy)+I*(wx*wy)
    return out

def sym_grad(u):
    """Compute the gradient of u with symmetric boundary conditions."""
    ux = np.zeros_like(u, dtype=np.float32)
    uy = np.zeros_like(u, dtype=np.float32)

    ux[:, :-1] = u[:, 1:] - u[:, :-1]
    uy[:-1, :] = u[1:, :] - u[:-1, :]

    return ux, uy

def sym_div(px: np.ndarray, py: np.ndarray) -> np.ndarray:
    """Divergence with forward differences and symmetric boundary handling.
    """
    div = np.zeros_like(px, dtype=np.float32)
    div[:, 0] += px[:, 0]
    div[:, 1:] += px[:, 1:] - px[:, :-1]
    div[0, :] += py[0, :]
    div[1:, :] += py[1:, :] - py[:-1, :]
    return div


def sym_index(k, N):
    k = k % (2*N)
    if k < N:
        return k
    else:
        return 2*N - 1 - k

def extended_sym(u):
    """Extend the image u with symmetric boundary conditions."""
    """Extend u to (2N,2M) by mirror symmetry (vectorized)."""
    left_right = np.concatenate((u, np.fliplr(u)), axis=1)
    return np.concatenate((left_right, np.flipud(left_right)), axis=0)



def d_prob(u, b, gamma):
    ux, uy = sym_grad(u)
    gx, gy = ux +b[0], uy + b[1]
    norm = np.sqrt(gx**2 + gy**2)
    norm_ = np.maximum(norm, 1e-8)  
    scale = np.maximum(0, norm - 1/gamma)/norm_
    d = (scale * gx, scale * gy)
    return (ux, uy), (d[0].astype(np.float32, copy=False), d[1].astype(np.float32, copy=False))


def u_prob(v, fft_K, d, b, lamb, gamma, denom, fft_vext):
    """Solve the u-subproblem using FFT.
    v: observed image
    K: convolution kernel (assumed to be centered and of same size as v_extended)
    """
    N, M = v.shape
    div_db = extended_sym(sym_div(d[0] - b[0], d[1] - b[1]))
    Fphi = fft_K
    num = (lamb / gamma) * (Fphi * fft_vext) - fft(div_db)

    u_new = np.real(ifft(num / denom))
    u_new = u_new[:N, :M]
    u_new = np.clip(u_new, 0, 1)
    return u_new


def extend_kernel(K, shape):
    """Extend the kernel K to the given shape with zero-padding."""
    N, M = shape
    K_ext = np.zeros((N, M), dtype=np.float32)
    kN, kM = K.shape
    K_ext[:kN, :kM] = K
    K_ext = np.roll(K_ext, -(K.shape[0]//2), axis=0)
    K_ext = np.roll(K_ext, -(K.shape[1]//2), axis=1)
    return K_ext/np.sum(K_ext)



def tv_deconv(v, K, lam=1000, gamma = 5, max_iters = 140, tol = None):
    
    N, M = v.shape
    u = np.zeros((N, M), dtype=np.float32)
    d = (np.zeros((N, M), dtype=np.float32), np.zeros((N, M), dtype=np.float32))
    b = (np.zeros((N, M), dtype=np.float32), np.zeros((N, M), dtype=np.float32))
    K_ = extend_kernel(K, (2*N, 2*M))
    fft_K = fft(K_)
    conj_fft_K = np.conj(fft_K)
    LAP = np.zeros((2*N, 2*M), dtype=np.float32)
    LAP[0, 0] = 4
    LAP[1, 0] = LAP[-1, 0] = LAP[0, 1] = LAP[0, -1] = -1
    FDelta = -np.real(fft(LAP))
    v_ext = extended_sym(v)
    fft_vext = fft(v_ext)
    denom = (lam / gamma) * (np.abs(fft_K) ** 2) - FDelta + 1e-8

    if tol is None:
        tol = norm(v_ext)/1e3
    print(tol)                                                      
    c=0
    for _ in range(max_iters):
        c += 1
        u_prev = u
        u_new = u_prob(v,conj_fft_K, d, b, lam, gamma, denom, fft_vext=fft_vext)
        (gx, gy), d_new = d_prob(u_new, b, gamma)
        b = (b[0] + (gx - d_new[0]),
             b[1] + (gy - d_new[1]))

        if norm(u_new - u_prev) < tol:
            break

        u = u_new
        d = d_new
    # print(c)
    return u, c  






# -------------------------------------------------------- Prof --------------------------------------------------------
# -------------------------------------------------------- Prof --------------------------------------------------------
# -------------------------------------------------------- Prof --------------------------------------------------------
# -------------------------------------------------------- Prof --------------------------------------------------------
# -------------------------------------------------------- Prof --------------------------------------------------------
# -------------------------------------------------------- Prof --------------------------------------------------------


#%% prof's TV deconvolution function for comparison
def pad_image(im,pad=10):
    out=np.zeros((im.shape[0]+2*pad,im.shape[1]+2*pad))
    out[pad:-pad,pad:-pad]=im
    for k in range(pad):
        out[k,pad:-pad]=im[0,:]
        out[-k-1,pad:-pad]=im[-1,:]
        out[pad:-pad,k]=im[:,0]
        out[pad:-pad,-k-1]=im[:,-1]
    out[:pad,:pad]=im[0,0]
    out[-pad:,:pad]=im[-1,0]
    out[:pad,-pad:]=im[0,-1]
    out[-pad:,-pad:]=im[-1,-1]
    return out

def unpad_image(im,pad=10):
    return im[pad:-pad,pad:-pad].copy()

def Fourier_kernel(K,s):
    assert K.shape[0]%2==1 and K.shape[1]%2==1, "Taille de noyau non impaire"
    Kf=np.zeros(s)
    Ky,Kx=K.shape
    Kx2=Kx//2
    Ky2=Ky//2
    Kf[:Ky2+1,:Kx2+1]=K[Ky2:,Kx2:]
    Kf[:Ky2+1,-Kx2:]=K[Ky2:,:Kx2]
    Kf[-Ky2:,:Kx2+1]=K[:Ky2,Kx2:]
    Kf[-Ky2:,-Kx2:]=K[:Ky2,:Kx2]
    return fft2(Kf)

def taper_image(I,K):
    """ Floute une image I par le noyau K (circulairement) cela donne une image J
    On mélange l'image I avec l'image J de manière à ce que J soit prépondérente aux bords.
    L'image J, lorsqu'on la déconle par le noyau K n'aura pas d'effets de bord. """
    kh,kw=K.shape
    Ih,Iw=I.shape
    wx=np.ones((Ih,Iw),dtype=np.float32)
    wy=np.ones((Ih,Iw),dtype=np.float32)
    X,Y=np.meshgrid(np.arange(0,Iw),np.arange(0,Ih))
    wy[:kh,:]=sin(Y[:kh,:]*pi/(2*kh-1))**2
    wy[-kh:,:]=sin((Ih-Y[-kh:,:])*pi/(2*kh-1))**2
    wx[:,:kw]=sin(X[:,:kw]*pi/(2*kh-1))**2
    wx[:,-kw:]=sin((Iw-X[:,-kw:])*pi/(2*kh-1))**2
    fK=Fourier_kernel(K,I.shape)
    J=real(ifft2(fft2(I)*fK))
    out=J*(1-wx*wy)+I*(wx*wy)
    return out

def conv(im,K,Fourierform=False):
    if not Fourierform:#on recoit les formes spatiales
        Kf=Fourier_kernel(K,im.shape)
        imf=fft2(im)
        return np.real(ifft2(imf*Kf))
    else:# forme Fourier
        return np.real(ifft2(im*K))

def champ_grad(u):#gradient circulaire
    return np.stack((np.c_[(u[:,0]-u[:,-1]).reshape(-1,1),u[:,1:]-u[:,:-1]],\
             np.r_[(u[0,:]-u[-1,:]).reshape(1,-1),u[1:,:]-u[:-1,:]]))

def universal_dot(X,Y):
    return (X*Y).sum()

def div_champ(c):
    return np.c_[c[0,:,1:]-c[0,:,:-1],(c[0,:,0]-c[0,:,-1]).reshape(-1,1)]+\
            np.r_[c[1,1:,:]-c[1,:-1,:],(c[1,0,:]-c[1,-1,:]).reshape(1,-1)]




def d_sub_problem(u,b,gamma=5/255):
    gradu=champ_grad(u)
    champ=gradu+b
    s=champ.shape[1:]
    no=(champ**2).sum(axis=0)**0.5
    mask=(no<(1/gamma))
    no[mask]=0.001
    no=no.reshape(1,*s)
    mu=1-1/(gamma*no)
    champ*=mu
    champ[:,mask]=0
    #champ[1,mask]=0
    return champ

def u_sub_problem(f,d,b,K,lamb,gamma=5,Fourierform=False,fdenom=None):
    """ si Fourierform=True alors f et K sont donnees sous forme Fourier"""
    if not Fourierform:
        ff=fft2(f)
        Kf=Fourier_kernel(K,f.shape)
    else:
        ff=f
        Kf=K
    if fdenom is None:
        Kl=np.zeros(f.shape)
        Kl[0,0]=4
        Kl[0,1]=-1
        Kl[1,0]=-1
        Kl[-1,0]=-1
        Kl[0,-1]=-1
        fdenom=real(fft2(Kl))
        fdenom+=(lamb/gamma)*(abs(Kf)**2)

    numer=conj(Kf)*ff*(lamb/gamma)-fft2(div_champ(d-b))
    return real(ifft2((numer)/fdenom))

def sym_image(x):
    out=np.concatenate((x,np.fliplr(x)),axis=1)
    out=np.concatenate((out,np.flipud(out)),axis=0) #symetrise l'image
    return out

def TV(im):
    g=champ_grad(im)
    n=((g**2).sum(axis=0))**0.5
    return n.sum()

def fonctionnelle(f,u,K,d,b,lamb,gamma=5):
    v1=TV(u)+lamb/2*norm(f-conv(f,K))**2
    v2=v1+gamma/2*norm(d-champ_grad(u)-b)**2
    return (v1,v2)

def norm(X):
    return ((X**2).sum())**0.5

def TVdeconv(im,K,lamb,nbit=140,gamma=5/255,edgehandle='taper'):
    """
    Si edgehandle= 'taper' alors on ajoute à l'image une bordure lisse
    Si edgehandle= 'sym' alors on symmetrise l'image
    Si edgehandle= 'nothing' alors on ne fait rien (mauvais)
    """
    if edgehandle=='taper':
        f=taper_image(pad_image(im,K.shape[0]),K)
    elif edgehandle=='sym':
        f=sym_image(im)
    else:
        f=im.copy()
    s=f.shape
    Kf=Fourier_kernel(K,s)
    Kl=np.zeros(f.shape)
    Kl[0,0]=4
    Kl[0,1]=-1 
    Kl[1,0]=-1
    Kl[-1,0]=-1
    Kl[0,-1]=-1
    fdenom=real(fft2(Kl))
    fdenom+=lamb/gamma*(abs(Kf)**2)
    u=np.zeros(s)
    unew=np.zeros(s)
    d=np.zeros((2,*s))
    b=np.zeros((2,*s))
    tol=norm(f)/1000
    counter=0
    ff=fft2(f)
    #Kfff=conj(Kf)*ff*(lamb/gamma)
    #print("iteration",counter,' Fonctionnelles=',\
     #         fonctionnelle(f,unew,K,d,b,lamb))

    while counter==0 or (norm(unew-u)>tol and counter<nbit):
        counter+=1
        u=unew
        d=d_sub_problem(u, b,gamma=gamma)

        unew=u_sub_problem(ff, d, b, Kf, lamb,gamma=gamma,\
                           Fourierform=True,fdenom=fdenom)
        b+=(champ_grad(unew)-d)
        #print("iteration",counter,)#' Fonctionnelles=',\
         #     fonctionnelle(f,unew,K,d,b,lamb))

    if edgehandle=='taper':
        out=unpad_image(unew,K.shape[0])
    elif edgehandle=='sym':
        out=unew[:im.shape[0],:im.shape[1]]
    else:
        out=unew
    print(counter)
    return out, counter



# -------------------------------------------------------- Utils --------------------------------------------------------
# -------------------------------------------------------- Utils --------------------------------------------------------
# -------------------------------------------------------- Utils --------------------------------------------------------
# -------------------------------------------------------- Utils --------------------------------------------------------
# -------------------------------------------------------- Utils --------------------------------------------------------
# -------------------------------------------------------- Utils --------------------------------------------------------
# -------------------------------------------------------- Utils --------------------------------------------------------


#%% Utilities for generating blur kernels and adding noise

# --- Helpers for random PSFs (Point Spread Functions) ---

def _normalize_kernel(K: np.ndarray) -> np.ndarray:
    K = K.astype(np.float32, copy=False)
    s = K.sum()
    if s <= 0:
        return K
    return K / s


def _center_kernel(K: np.ndarray) -> np.ndarray:
    """Circularly shift so that the mass is centered at (size//2, size//2)."""
    cy, cx = np.array(K.shape) // 2
    # Find argmax as a coarse center and roll to the middle
    iy, ix = np.unravel_index(np.argmax(K), K.shape)
    K = np.roll(K, cy - iy, axis=0)
    K = np.roll(K, cx - ix, axis=1)
    return K


def gaussian_kernel(size=11, sigma_x=2.0, sigma_y=None, theta=0.0):
    """Anisotropic (rotated) Gaussian kernel."""
    if sigma_y is None:
        sigma_y = sigma_x
    ax = np.linspace(-(size // 2), size // 2, size, dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax)
    c, s = np.cos(theta), np.sin(theta)
    xr = c * xx + s * yy
    yr = -s * xx + c * yy
    K = np.exp(-0.5 * ((xr / sigma_x) ** 2 + (yr / sigma_y) ** 2))
    return _normalize_kernel(K)


def box_kernel(size=11):
    K = np.ones((size, size), dtype=np.float32)
    return _normalize_kernel(K)


def disk_kernel(size=11, radius=None):
    if radius is None:
        radius = (size - 1) / 2 * 0.7
    ax = np.linspace(-(size // 2), size // 2, size, dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax)
    K = (xx ** 2 + yy ** 2) <= radius ** 2
    return _normalize_kernel(K.astype(np.float32))


def motion_kernel(size=11, length=None, angle=0.0):
    """Simple line-motion blur with anti-aliased line using a Gaussian along the path."""
    if length is None:
        length = max(3, int(0.6 * size))
    K = np.zeros((size, size), dtype=np.float32)
    c, s = np.cos(angle), np.sin(angle)
    cy, cx = (size - 1) / 2, (size - 1) / 2
    half = (length - 1) / 2
    t = np.linspace(-half, half, length)
    xs = cx + c * t
    ys = cy + s * t
    for x, y in zip(xs, ys):
        ix, iy = int(round(x)), int(round(y))
        if 0 <= iy < size and 0 <= ix < size:
            K[iy, ix] += 1.0
    # Slightly blur the discrete line to anti-alias
    K = scipy.ndimage.gaussian_filter(K, sigma=0.5)
    return _normalize_kernel(K)


def random_walk_kernel(size=11, steps=50, seed=None):
    """Kernel from a 2D random walk path (camera shake style)."""
    rng = np.random.default_rng(seed)
    K = np.zeros((size, size), dtype=np.float32)
    y, x = (size - 1) / 2, (size - 1) / 2
    for _ in range(steps):
        dy, dx = rng.normal(0, 1, size=2)
        y = np.clip(y + dy, 0, size - 1)
        x = np.clip(x + dx, 0, size - 1)
        K[int(round(y)), int(round(x))] += 1.0
    K = scipy.ndimage.gaussian_filter(K, sigma=0.8)
    return _normalize_kernel(K)


def mog_kernel(size=11, n_comp=2, sigma_range=(1.0, 3.0), seed=None):
    """Mixture of Gaussians randomly placed."""
    rng = np.random.default_rng(seed)
    K = np.zeros((size, size), dtype=np.float32)
    for _ in range(n_comp):
        sx = rng.uniform(*sigma_range)
        sy = rng.uniform(*sigma_range)
        theta = rng.uniform(0, np.pi)
        # Random center
        cy = rng.integers(low=size // 4, high=3 * size // 4)
        cx = rng.integers(low=size // 4, high=3 * size // 4)
        ax = np.arange(size, dtype=np.float32)
        xx, yy = np.meshgrid(ax, ax)
        xx = xx - cx
        yy = yy - cy
        c, s = np.cos(theta), np.sin(theta)
        xr = c * xx + s * yy
        yr = -s * xx + c * yy
        K += np.exp(-0.5 * ((xr / sx) ** 2 + (yr / sy) ** 2))
    K = _normalize_kernel(K)
    return _center_kernel(K)


def random_blur_kernel(size=11, sigma_range=(1.5, 3.0), seed=None, id=0):
    """
    Generate various random blur kernels (PSFs) by id:
      id=0: Identity
      id=1: "prof" custom kernel (as in original code)
      id=2: Isotropic Gaussian with sigma ~ U(sigma_range)
      id=3: Anisotropic Gaussian with random sigma_x, sigma_y, theta
      id=4: Motion blur with random length and angle
      id=5: Disk (defocus) with random radius
      id=6: Box (average) filter
      id=7: Mixture of Gaussians (random components)
      id=8: Random-walk (camera shake) kernel
    """
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

    if id == 0:  # Identity kernel
        K = np.zeros((size, size), dtype=np.float32)
        K[size // 2, size // 2] = 1.0
        return K

    if id == 1:  # prof kernel (kept from original)
        K = np.zeros((size, size), dtype=np.float32)
        for k in range(size):
            K[k, k] = 1
            if k > 0:
                K[k, k - 1] = 1
            if k < size - 1:
                K[k, k + 1] = 1
        return _normalize_kernel(K)

    if id == 2:  # Isotropic Gaussian
        sigma = rng.uniform(*sigma_range)
        return gaussian_kernel(size=size, sigma_x=sigma, sigma_y=sigma, theta=0.0)

    if id == 3:  # Anisotropic Gaussian
        sx = rng.uniform(*sigma_range)
        sy = rng.uniform(*sigma_range)
        th = rng.uniform(0, np.pi)
        return gaussian_kernel(size=size, sigma_x=sx, sigma_y=sy, theta=th)

    if id == 4:  # Motion blur
        length = size
        angle = rng.uniform(0, np.pi)
        return motion_kernel(size=size, length=int(length), angle=angle)

    if id == 5:  # Disk (defocus)
        radius = rng.uniform((size - 1) / 4, (size - 1) / 2)
        return disk_kernel(size=size, radius=radius)

    if id == 6:  # Box
        return box_kernel(size=size)

    if id == 7:  # Mixture of Gaussians
        n_comp = int(rng.integers(low=2, high=4))
        return mog_kernel(size=size, n_comp=n_comp, sigma_range=sigma_range, seed=seed)

    if id == 8:  # Random walk (camera shake)
        steps = int(rng.integers(low=20, high=80))
        return random_walk_kernel(size=size, steps=steps, seed=seed)

    # Fallback: isotropic Gaussian
    sigma = rng.uniform(*sigma_range)
    return gaussian_kernel(size=size, sigma_x=sigma, sigma_y=sigma, theta=0.0)


def add_gaussian_noise(img, sigma=0.01, seed=None, clip=True):
    if seed is not None:
        np.random.seed(seed)
    noise = sigma * np.random.randn(*img.shape).astype(np.float32)
    noisy = img.astype(np.float32) + noise
    if clip:
        noisy = np.clip(noisy, 0.0, 1.0)
    return noisy



def snr(x, xhat):
    num = np.sum(x**2)
    den = np.sum((x - xhat)**2) + 1e-12
    return 10.0 * np.log10(num / den)


# -------------------------------------------------------- Test --------------------------------------------------------
# -------------------------------------------------------- Test --------------------------------------------------------
# -------------------------------------------------------- Test --------------------------------------------------------


# %% Basic test 
image = load_image('lena.tif')

K = random_blur_kernel(size=11, sigma_range=(1.0, 3.0), seed=None, id=1)
print(K.shape)
v = convolve2d(image, K, mode='same', boundary='symm')
view_image(image, v, K, titles=['Original', 'Blurred', 'Kernel'])

start = time.perf_counter()
u_perso, _ = tv_deconv(v, K, lam=1000, gamma=5, max_iters=140)
end = time.perf_counter()
print(f"Computation time (mon algo): {end - start:.2f} seconds")
start = time.perf_counter()
u_prof, _ = TVdeconv(v, K, lamb=1000, nbit=140, gamma=5, edgehandle='sym')
end = time.perf_counter()
print(f"Computation time (algo du prof): {end - start:.2f} seconds")

view_image(u_perso, u_prof, titles=['My TV Deconvolution', "Prof's TV Deconvolution"])

print(f"SNR of my solution: {snr(image, u_perso):.2f} dB")
print(f"SNR of prof's solution: {snr(image, u_prof):.2f} dB")



# -------------------------------------------------------- Benchmark --------------------------------------------------------
# -------------------------------------------------------- Benchmark --------------------------------------------------------
# -------------------------------------------------------- Benchmark --------------------------------------------------------
# %%

def benchmarks_test(test_hyperparam=True, test_kernels= True):
    image = load_image('lena.tif')
    if test_hyperparam:
        K = random_blur_kernel(size=11, sigma_range=(1.0, 3.0), seed=0, id=1, )
        v = convolve2d(image, K, mode='same', boundary='symm')

        view_image(image, v, K, titles=['Original', 'Blurred', 'Kernel'])

        
        L = [1, 100, 1000, 5000, 10000]
        G = [0.1, 5, 10, 20, 100]

        H, W = image.shape
        IL = np.zeros((H, W, len(L), 2), dtype=np.float32) 
        IG = np.zeros((H, W, len(G), 2), dtype=np.float32)

        PL = []
        PG = []  

        # Balayage de lambda
        for i, lam in enumerate(L):
            print(f"Test Lambda = {lam}")

            t0 = time.perf_counter()
            u_perso, nb_iter1 = tv_deconv(v, K, lam=lam, gamma=5, max_iters=140)
            dt1 = time.perf_counter() - t0
            snr1 = snr(image, u_perso)

            t0 = time.perf_counter()
            u_prof, nb_iter2 = TVdeconv(v, K, lamb=lam, gamma=5, nbit=140, edgehandle='sym')
            dt2 = time.perf_counter() - t0
            snr2 = snr(image, u_prof)

            IL[:, :, i, 0] = u_perso
            IL[:, :, i, 1] = u_prof
            PL.append([(nb_iter1, dt1, snr1), (nb_iter2, dt2, snr2)])

        # Balayage de gamma
        for j, gam in enumerate(G):
            print(f"Test Gamma = {gam}")

            t0 = time.perf_counter()
            u_perso, nb_iter1 = tv_deconv(v, K, lam=1000, gamma=gam, max_iters=140)
            dt1 = time.perf_counter() - t0
            snr1 = snr(image, u_perso)

            t0 = time.perf_counter()
            u_prof, nb_iter2 = TVdeconv(v, K, lamb=1000, gamma=gam, nbit=140, edgehandle='sym')
            dt2 = time.perf_counter() - t0
            snr2 = snr(image, u_prof)

            IG[:, :, j, 0] = u_perso
            IG[:, :, j, 1] = u_prof
            PG.append([(nb_iter1, dt1, snr1), (nb_iter2, dt2, snr2)])

        # --- Pour lambda ---
        imgs_my_lambda = []
        titles_my_lambda = []
        imgs_prof_lambda = []
        titles_prof_lambda = []
        for i, lam in enumerate(L):
            imgs_my_lambda.append(IL[:, :, i, 0])
            titles_my_lambda.append(f"My TV (λ={lam}, SNR={PL[i][0][2]:.2f} dB)")
            imgs_prof_lambda.append(IL[:, :, i, 1])
            titles_prof_lambda.append(f"Prof TV (λ={lam}, SNR={PL[i][1][2]:.2f} dB)")

        view_image(*imgs_my_lambda, titles=titles_my_lambda)
        view_image(*imgs_prof_lambda, titles=titles_prof_lambda)

        # --- Pour gamma ---
        imgs_my_gamma = []
        titles_my_gamma = []
        imgs_prof_gamma = []
        titles_prof_gamma = []
        for j, gam in enumerate(G):
            imgs_my_gamma.append(IG[:, :, j, 0])
            titles_my_gamma.append(f"My TV (γ={gam}, SNR={PG[j][0][2]:.2f} dB)")
            imgs_prof_gamma.append(IG[:, :, j, 1])
            titles_prof_gamma.append(f"Prof TV (γ={gam}, SNR={PG[j][1][2]:.2f} dB)")

        view_image(*imgs_my_gamma, titles=titles_my_gamma)
        view_image(*imgs_prof_gamma, titles=titles_prof_gamma)

        fig, axes = plt.subplots(3, 2, figsize=(10, 10), dpi=300, constrained_layout=True)

        # --- λ: Iterations, Time, SNR ---
        ax = axes[0, 0]
        ax.plot(L, [p[0][0] for p in PL], marker='o', label='My iters')
        ax.plot(L, [p[1][0] for p in PL], marker='o', label='Prof iters')
        ax.set_xlabel('Lambda'); ax.set_ylabel('Iterations'); ax.set_title('Iterations vs λ'); ax.legend()

        ax = axes[1, 0]
        ax.plot(L, [p[0][1] for p in PL], marker='o', label='My time (s)')
        ax.plot(L, [p[1][1] for p in PL], marker='o', label='Prof time (s)')
        ax.set_xlabel('Lambda'); ax.set_ylabel('Seconds'); ax.set_title('Time vs λ'); ax.legend()

        ax = axes[2, 0]
        ax.plot(L, [p[0][2] for p in PL], marker='o', label='My SNR (dB)')
        ax.plot(L, [p[1][2] for p in PL], marker='o', label='Prof SNR (dB)')
        ax.set_xlabel('Lambda'); ax.set_ylabel('SNR (dB)'); ax.set_title('SNR vs λ'); ax.legend()

        # --- γ: Iterations, Time, SNR ---
        ax = axes[0, 1]
        ax.plot(G, [p[0][0] for p in PG], marker='o', label='My iters')
        ax.plot(G, [p[1][0] for p in PG], marker='o', label='Prof iters')
        ax.set_xlabel('Gamma'); ax.set_ylabel('Iterations'); ax.set_title('Iterations vs γ'); ax.legend()

        ax = axes[1, 1]
        ax.plot(G, [p[0][1] for p in PG], marker='o', label='My time (s)')
        ax.plot(G, [p[1][1] for p in PG], marker='o', label='Prof time (s)')
        ax.set_xlabel('Gamma'); ax.set_ylabel('Seconds'); ax.set_title('Time vs γ'); ax.legend()

        ax = axes[2, 1]
        ax.plot(G, [p[0][2] for p in PG], marker='o', label='My SNR (dB)')
        ax.plot(G, [p[1][2] for p in PG], marker='o', label='Prof SNR (dB)')
        ax.set_xlabel('Gamma'); ax.set_ylabel('SNR (dB)'); ax.set_title('SNR vs γ'); ax.legend()

        plt.show()


    # --- Comparaison sur différents noyaux (λ=1000, γ=5) ---
    if test_kernels:
        kernel_specs = [
            {"id": 1, "name": "prof"},
            {"id": 2, "name": "gauss iso"},
            {"id": 4, "name": "motion"},
            {"id": 7, "name": "random"},
            {"id": 8, "name": "random walk"},
        ]
        lam_fixed = 1000
        gamma_fixed = 5
        size = 11

        results = []  

        for k_idx, spec in enumerate(kernel_specs):
            K = random_blur_kernel(size=size, sigma_range=(1.0, 3.0), seed=None, id=spec["id"]) 
            v = convolve2d(np.asarray(image, dtype=np.float32), np.asarray(K, dtype=np.float32), mode='same', boundary='symm')
            t0 = time.perf_counter()
            u_my, it_my = tv_deconv(v, K, lam=lam_fixed, gamma=gamma_fixed, max_iters=140)
            t_my = time.perf_counter() - t0
            snr_my = snr(image, u_my)
            t0 = time.perf_counter()
            u_prof, it_prof = TVdeconv(v, K, lamb=lam_fixed, gamma=gamma_fixed, nbit=140, edgehandle='sym')
            t_prof = time.perf_counter() - t0
            snr_prof = snr(image, u_prof)
            results.append((spec["name"], it_my, t_my, snr_my, it_prof, t_prof, snr_prof))
            view_image(
                image,
                v,
                u_my,
                u_prof,
                K,
                titles=[
                    'Original',
                    f'Blurred (id={spec["id"]})',
                    f'My (SNR={snr_my:.2f} dB)',
                    f'Prof (SNR={snr_prof:.2f} dB)',
                    'Kernel'
                ],
                cmap='gray'
            )
        print("\nSummary over 5 kernels (λ=1000, γ=5):")
        print("Kernel\t\tMy it\tMy time(s)\tMy SNR(dB)\tProf it\tProf time(s)\tProf SNR(dB)")
        for (name, it_my, t_my, snr_my, it_prof, t_prof, snr_prof) in results:
            print(f"{name}\t\t{it_my}\t{t_my:.2f}\t{snr_my:.2f}\t{it_prof}\t{t_prof:.2f}\t{snr_prof:.2f}")




benchmarks_test(test_hyperparam=False, test_kernels=True)




# %%

