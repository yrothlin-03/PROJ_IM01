import numpy as np
from scipy.fft import fft2, ifft2
from scipy.signal import convolve2d

real = np.real
pi = np.pi
sin = np.sin


def pad_image(im, pad=10):
    H, W = im.shape
    out = np.zeros((H + 2 * pad, W + 2 * pad), dtype=im.dtype)
    out[pad:-pad, pad:-pad] = im

    for k in range(pad):
        out[k, pad:-pad] = im[0, :]
        out[-k - 1, pad:-pad] = im[-1, :]
        out[pad:-pad, k] = im[:, 0]
        out[pad:-pad, -k - 1] = im[:, -1]

    out[:pad, :pad] = im[0, 0]
    out[-pad:, :pad] = im[-1, 0]
    out[:pad, -pad:] = im[0, -1]
    out[-pad:, -pad:] = im[-1, -1]

    return out


def unpad_image(im, pad=10):
    return im[pad:-pad, pad:-pad].copy()


def fourier_kernel(K, shape):
    Kh, Kw = K.shape
    H, W = shape
    assert Kh % 2 == 1 and Kw % 2 == 1, "K.shape must be odd."

    Kf = np.zeros(shape, dtype=np.float32)

    ky2 = Kh // 2
    kx2 = Kw // 2

    Kf[:ky2 + 1, :kx2 + 1] = K[ky2:, kx2:]
    Kf[:ky2 + 1, -kx2:] = K[ky2:, :kx2]
    Kf[-ky2:, :kx2 + 1] = K[:ky2, kx2:]
    Kf[-ky2:, -kx2:] = K[:ky2, :kx2]

    return fft2(Kf)


def taper_image(I, K):
    kh, kw = K.shape
    Ih, Iw = I.shape

    wx = np.ones((Ih, Iw), dtype=np.float32)
    wy = np.ones((Ih, Iw), dtype=np.float32)

    X, Y = np.meshgrid(np.arange(Iw), np.arange(Ih))

    if kh > 0 and kh < Ih:
        wy[:kh, :] = sin(Y[:kh, :] * pi / (2 * kh - 1)) ** 2
        wy[-kh:, :] = sin((Ih - 1 - Y[-kh:, :]) * pi / (2 * kh - 1)) ** 2

    if kw > 0 and kw < Iw:
        wx[:, :kw] = sin(X[:, :kw] * pi / (2 * kw - 1)) ** 2
        wx[:, -kw:] = sin((Iw - 1 - X[:, -kw:]) * pi / (2 * kw - 1)) ** 2

    w = wx * wy

    Kf = fourier_kernel(K, I.shape)
    J = real(ifft2(fft2(I) * Kf))

    out = J * (1 - w) + I * w
    return out.astype(np.float32, copy=False)


def grad_circ(u):
    gx = np.roll(u, -1, axis=1) - u
    gy = np.roll(u, -1, axis=0) - u
    return gx, gy


def div_circ(px, py):
    dx = px - np.roll(px, 1, axis=1)
    dy = py - np.roll(py, 1, axis=0)
    return dx + dy


def d_sub_problem(u, b, gamma):
    gx, gy = grad_circ(u)
    cx = gx + b[0]
    cy = gy + b[1]

    norm_c = np.sqrt(cx ** 2 + cy ** 2)
    norm_c_safe = np.maximum(norm_c, 1e-8)

    factor = np.maximum(0.0, 1.0 - 1.0 / (gamma * norm_c_safe))
    dx = factor * cx
    dy = factor * cy

    mask = norm_c < (1.0 / gamma)
    dx[mask] = 0.0
    dy[mask] = 0.0

    return (dx, dy)


def u_sub_problem(f, d, b, Kf, lamb, gamma, fdenom):
    gx, gy = d
    bx, by = b
    div_db = div_circ(gx - bx, gy - by)

    Ff = fft2(f)
    rhs = (lamb / gamma) * np.conj(Kf) * Ff - fft2(div_db)

    U = rhs / fdenom
    u = real(ifft2(U))
    return u.astype(np.float32, copy=False)


def tv_deconv_circular(im, K, lam=1472, n_iter=140, gamma=13, add_tapping=False):
    im = im.astype(np.float32, copy=False)

    if add_tapping:
        f = taper_image(pad_image(im, K.shape[0]), K)
    else:
        f = im.copy()

    H, W = f.shape

    Kf = fourier_kernel(K, f.shape)

    Kl = np.zeros_like(f, dtype=np.float32)
    Kl[0, 0] = 4.0
    Kl[0, 1] = -1.0
    Kl[1, 0] = -1.0
    Kl[-1, 0] = -1.0
    Kl[0, -1] = -1.0

    fdenom = real(fft2(Kl)) + (lam / gamma) * (np.abs(Kf) ** 2)

    u = np.zeros_like(f, dtype=np.float32)
    d = (np.zeros_like(f, dtype=np.float32),
         np.zeros_like(f, dtype=np.float32))
    b = (np.zeros_like(f, dtype=np.float32),
         np.zeros_like(f, dtype=np.float32))

    if im.size == 0:
        tol = 0.0
    else:
        tol = np.linalg.norm(f) / 1000.0

    for it in range(n_iter):
        u_old = u.copy()

        d = d_sub_problem(u, b, gamma=gamma)

        u = u_sub_problem(f, d, b, Kf, lam, gamma, fdenom)

        gx, gy = grad_circ(u)
        bx, by = b
        b = (bx + (gx - d[0]),
             by + (gy - d[1]))

        if np.linalg.norm(u - u_old) < tol:
            break

    if add_tapping:
        out = unpad_image(u, K.shape[0])
    else:
        out = u

    return out