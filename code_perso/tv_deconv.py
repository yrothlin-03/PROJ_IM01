import numpy as np
from scipy.signal import convolve2d
import scipy


fft = scipy.fft.rfft2
ifft = scipy.fft.irfft2
norm = np.linalg.norm

real = np.real
sin = np.sin
conj = np.conj
pi = np.pi



def pad_images(img, pad_width):
    if img.ndim == 3 and img.shape[2] in (1, 3):
        H, W = img.shape[:2]
    else:
        H, W = img.shape
    padded_img = np.pad(img, ((pad_width, pad_width), (pad_width, pad_width), (0, 0)) if img.ndim == 3 else ((pad_width, pad_width), (pad_width, pad_width)), mode='symmetric')
    return padded_img


def unpad_images(padded_img, pad_width):
    if padded_img.ndim == 3 and padded_img.shape[2] in (1, 3):
        return padded_img[pad_width:-pad_width, pad_width:-pad_width, :]
    else:
        return padded_img[pad_width:-pad_width, pad_width:-pad_width]


def tukey_window(n, alpha=0.5):
    w = np.ones(n)
    edge = int(alpha * n / 2)
    if edge > 0:
        t = np.linspace(0, np.pi, edge)
        w[:edge] = 0.5 * (1 - np.cos(t))
        w[-edge:] = 0.5 * (1 - np.cos(t[::-1]))
    return w

def tapper(img, kernel=None, margin=None):
    if img.ndim == 3 and img.shape[2] in (1, 3):
        H, W = img.shape[:2]
    else:
        H, W = img.shape

    if margin is None:
        margin = max(8, min(H, W) // 10)

    wy = tukey_window(H, alpha=margin / H)
    wx = tukey_window(W, alpha=margin / W)
    window = np.outer(wy, wx).astype(img.dtype, copy=False)


    if kernel is None:
        return (window[..., None] * img) if img.ndim == 3 else (window * img)


    if img.ndim == 2:
        blurred = convolve2d(img, kernel, mode="same", boundary="symm")
        return window * img + (1.0 - window) * blurred
    else:

        out = np.empty_like(img)
        for c in range(img.shape[2]):
            blurred_c = convolve2d(img[:, :, c], kernel, mode="same", boundary="symm")
            out[:, :, c] = window * img[:, :, c] + (1.0 - window) * blurred_c
        return out


def sym_grad(u):
    ux = np.zeros_like(u, dtype=np.float32)
    uy = np.zeros_like(u, dtype=np.float32)

    ux[:, :-1] = u[:, 1:] - u[:, :-1]
    uy[:-1, :] = u[1:, :] - u[:-1, :]

    return ux, uy

def sym_div(px: np.ndarray, py: np.ndarray) -> np.ndarray:
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
    N, M = v.shape
    div_db = extended_sym(sym_div(d[0] - b[0], d[1] - b[1]))
    Fphi = fft_K
    num = (lamb / gamma) * (Fphi * fft_vext) - fft(div_db)

    u_new = np.real(ifft(num / denom))
    u_new = u_new[:N, :M]
    u_new = np.clip(u_new, 0, 1)
    return u_new


def extend_kernel(K, shape):
    N, M = shape
    K_ext = np.zeros((N, M), dtype=np.float32)
    kN, kM = K.shape
    K_ext[:kN, :kM] = K
    K_ext = np.roll(K_ext, -(K.shape[0]//2), axis=0)
    K_ext = np.roll(K_ext, -(K.shape[1]//2), axis=1)
    return K_ext/np.sum(K_ext)



def tv_deconv(v, K, lam=1472, gamma = 13, max_iters = 140, tol = None, add_tapping = True):

    if add_tapping:
        v = pad_images(v, pad_width=K.shape[0]).astype(np.float32, copy=False)
        v_in = tapper(v, K)
    else:
        v_in = v

    N, M = v_in.shape
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
    v_ext = extended_sym(v_in)
    fft_vext = fft(v_ext)
    denom = (lam / gamma) * (np.abs(fft_K) ** 2) - FDelta + 1e-8

    if tol is None:
        tol = norm(v_ext)/1e3                                                   
    c=0
    for _ in range(max_iters):
        c += 1
        u_prev = u
        u_new = u_prob(v_in,conj_fft_K, d, b, lam, gamma, denom, fft_vext=fft_vext)
        (gx, gy), d_new = d_prob(u_new, b, gamma)
        b = (b[0] + (gx - d_new[0]),
             b[1] + (gy - d_new[1]))

        if norm(u_new - u_prev) < tol:
            break

        u = u_new
        d = d_new
    if add_tapping:
        u = unpad_images(u, pad_width=K.shape[0])
    return u
