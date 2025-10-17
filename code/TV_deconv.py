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
    # print(tol)                                                      
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
