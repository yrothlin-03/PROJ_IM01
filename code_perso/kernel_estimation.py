import numpy as np
from tv_deconv import tv_deconv
import scipy
import scipy.linalg as la
import scipy.ndimage
from scipy.signal import convolve, fftconvolve
from scipy.sparse.linalg import cg, LinearOperator
import matplotlib.pyplot as plt
from kernel_estimation_bis import compute_R



def show_autocorr(R: np.ndarray):
    plt.figure()
    n_angles, win_len = R.shape
    center = win_len // 2
    for i in range(0, n_angles, 50):
        plt.plot(np.arange(-center, center + 1), R[i, :])

    plt.title("Projections' Autocorrelations")
    plt.xlabel("Lag")
    plt.ylabel("Autocorrelation")
    plt.legend()
    plt.savefig(f"kernel_est/autocorr.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


def show_autocorr_compensated(R_comp: np.ndarray):
    plt.figure()
    n_angles, win_len = R_comp.shape
    center = win_len // 2
    for i in range(0, n_angles, 50):
        plt.plot(np.arange(-center, center + 1), R_comp[i, :])

    plt.title("Compensated Projections' Autocorrelations")
    plt.xlabel("Lag")
    plt.ylabel("Autocorrelation")
    plt.legend()
    plt.savefig(f"kernel_est/autocorr_compensated.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


def show_whitening(Dtheta, theta):
    plt.figure()
    plt.imshow(Dtheta, cmap='gray', aspect='auto')
    plt.title(f"Whitening Matrix Dtheta for angle {theta}")
    plt.xlabel("Theta Index")
    plt.ylabel("Frequency Index")
    plt.colorbar()
    plt.savefig(f"kernel_est/whitening_matrix_theta_{theta}.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


    
def show_initial_support(S, vals_min=None):
    plt.figure()
    plt.plot(S.reshape(-1), label="Estimated Support", color='blue', linewidth=2)
    if vals_min is not None:
        plt.plot(vals_min.reshape(-1), label="Minimum Values", color='red', linestyle='--')
    plt.title("Initial Support")
    plt.savefig(f"kernel_est/initial_support.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

def show_power_spectrum(Hspectrum, iteration):
    plt.figure()
    plt.imshow(Hspectrum, cmap='hot', aspect='auto')
    plt.title(f"Estimated Power Spectrum of the Blur Kernel - Iteration {iteration}")
    plt.xlabel("Frequency X")
    plt.ylabel("Frequency Y")
    plt.colorbar()
    plt.savefig(f"kernel_est/power_spectrum_iteration_{iteration}.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

def show_kernel(kernel, iteration):
    plt.figure()
    plt.imshow(kernel, cmap='gray')
    plt.title(f"Estimated Blur Kernel - Iteration {iteration}")
    plt.colorbar()
    plt.savefig(f"kernel_est/kernel_iteration_{iteration}.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

def show_reestimated_support(supports, iteration):
    plt.figure()
    plt.plot(supports)
    plt.title(f"Re-estimated Supports of the Blur Kernel - Iteration {iteration}")
    plt.xlabel("Angle Index")
    plt.ylabel("Support Size")
    plt.savefig(f"kernel_est/reestimated_support_iteration_{iteration}.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


def _score_kernel(hk: np.ndarray, P: np.ndarray) -> float:
    d, _ = tv_deconv(P, hk)
    gx, gy = _grad(d)
    grad = np.sqrt(gx * gx + gy * gy)
    l1 = float(np.sum(np.abs(grad)))
    l2 = float(np.sqrt(np.sum(grad * grad) + 1e-12))
    return l1 / (l2 + 1e-12)

# def _compensation_filter(Mh: int, alph: float = 1.0, r: int = 4) -> np.ndarray:
#     half = (r * Mh) // 2
#     K = np.arange(-half, half + 1, dtype=np.int32)
#     L = (np.abs(K) + 1.0) ** (-alph)
#     H = L / L.sum()
#     return H.astype(np.float32)

def _grad(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gx = np.diff(img, axis=1, append=img[:, -1:])
    gy = np.diff(img, axis=0, append=img[-1:, :])
    return gx.astype(np.float32), gy.astype(np.float32)

def _extract_patch(img: np.ndarray, size: int = 150) -> np.ndarray:
    H, W = img.shape[:2]
    ph = min(size, H)
    pw = min(size, W)
    y0 = (H - ph) // 2
    x0 = (W - pw) // 2
    patch = img[y0:y0 + ph, x0:x0 + pw]
    return patch.astype(np.float32) 

def _autocorr_1d(q: np.ndarray, Mh: int, r: int = 4) -> np.ndarray:
    ac = np.correlate(q, q, mode='full').astype(np.float32)
    center = len(ac) // 2
    half = (r * Mh) // 2
    return ac[center - half:center + half + 1]



def _shear_projection(vx: np.ndarray, vy: np.ndarray, angle: float) -> np.ndarray:
    cos_t = np.cos(angle)
    sin_t = np.sin(angle)
    tan_t = np.tan(angle)

    H, W = vx.shape
    ys, xs = np.indices((H, W))
    xs = xs - W // 2
    ys = ys - H // 2

    w = vx * cos_t + vy * sin_t

    if np.abs(tan_t) <= 1.0:
        offset = xs + ys * tan_t
    else:
        offset = ys + xs / (tan_t + 1e-8)

    
    offset_flat = offset.ravel().astype(np.float64)
    w_flat = w.ravel().astype(np.float64)

    mask = np.isfinite(offset_flat)
    offset_flat = offset_flat[mask]
    w_flat = w_flat[mask]

    if offset_flat.size == 0:
        return np.zeros(1, dtype=np.float32)

    o_min = np.floor(offset_flat.min())
    o_max = np.ceil(offset_flat.max())
    size = int(o_max - o_min + 1)
    if size <= 0:
        return np.zeros(1, dtype=np.float32)

    idx = np.round(offset_flat - o_min).astype(np.int64)
    idx = np.clip(idx, 0, size - 1)

    q = np.bincount(idx, weights=w_flat, minlength=size).astype(np.float32)

    return q


def _shear_projection_kernel(v: np.ndarray, angle: float) -> np.ndarray:
    cos_t = np.cos(angle)
    sin_t = np.sin(angle)
    tan_t = np.tan(angle)

    H, W = v.shape
    ys, xs = np.indices((H, W))
    xs = xs - W // 2
    ys = ys - H // 2

    w = v.astype(np.float32)

    if np.abs(tan_t) <= 1.0:
        offset = xs + ys * tan_t
    else:
        offset = ys + xs / (tan_t + 1e-8)

    offset_flat = offset.ravel().astype(np.float64)
    w_flat = w.ravel().astype(np.float64)

    mask = np.isfinite(offset_flat)
    offset_flat = offset_flat[mask]
    w_flat = w_flat[mask]

    if offset_flat.size == 0:
        return np.zeros(1, dtype=np.float32)

    o_min = np.floor(offset_flat.min())
    o_max = np.ceil(offset_flat.max())
    size = int(o_max - o_min + 1)
    if size <= 0:
        return np.zeros(1, dtype=np.float32)

    idx = np.round(offset_flat - o_min).astype(np.int64)
    idx = np.clip(idx, 0, size - 1)

    q = np.bincount(idx, weights=w_flat, minlength=size).astype(np.float32)
    return q


def _deconv_1d_cg(y: np.ndarray, alph: float = 2.1, lam: float = 1e-2) -> np.ndarray:
    y = y.astype(np.float32)
    qp1 = int(y.shape[0])

    # We use Toeplitz M[k,l] = 1 / (|k-l|+1)**alph matrix for deconv : (M+lam I)x = y
    idx = np.abs(np.arange(qp1)[:, None] - np.arange(qp1)[None, :]).astype(np.float32)
    M = 1.0 / ((idx + 1.0) ** alph)
    M /= M[0, :].sum()

    A = M + lam * np.eye(qp1, dtype=np.float32)
    x = la.solve(A, y, assume_a='pos', check_finite=False)

    poscentre = qp1 // 2
    left = max(poscentre - 2, 0)
    right = min(poscentre + 2, qp1 - 1)
    window = x[left:right + 1]

    if window.min() < 0:
        return y

    x = np.maximum(x, 0)
    return x.astype(np.float32)


def center_kernel(h: np.ndarray) -> np.ndarray:
    p = h.shape[0] // 2
    cy, cx = np.unravel_index(np.argmax(h), h.shape)
    shift_y = p - cy
    shift_x = p - cx
    return np.roll(np.roll(h, shift_y, axis=0), shift_x, axis=1)


def ComputeProjectionAngleSet(Mh: int, r: int = 4) -> np.ndarray:
    A = set()
    half = (r * Mh) // 2
    for i in range(-half, half + 1):
        for j in range(-half, half + 1):
            if i == 0 and j == 0:
                continue
            if j < 0:
                continue
            if j == 0 and i < 0:
                continue
            if np.gcd(i, j) != 1:
                continue

            angle = np.arctan2(i, j)
            A.add(angle)

    return np.array(sorted(A, reverse=True), dtype=np.float32)


def ComputeProjectionsAutocorrelation(v: np.ndarray, AngleSet: np.ndarray, Mh: int, alph: float, r: int = 4, verbose: bool = False) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    if v.ndim == 3:
        v = v.mean(axis=2)
    N, M = v.shape[:2]

    d = np.array([3, -32, 168, -672, 0, 672, -168, 32, -3], dtype=np.float32) / 840.0

    vx = scipy.ndimage.convolve1d(v, d, axis=1, mode='reflect')
    vy = scipy.ndimage.convolve1d(v, d, axis=0, mode='reflect')

    win_len = r * Mh + 1
    R = np.zeros((len(AngleSet), win_len), dtype=np.float32)
    if verbose:
        print("Computing projections' autocorrelations.")
    for i, angle in enumerate(AngleSet):

        q = _shear_projection(vx, vy, angle)

        ac = _autocorr_1d(q, Mh, r=r)

        ac_deconv = _deconv_1d_cg(ac, alph=alph, lam=1e-2)

        R[i, :] = ac_deconv

    return R

def ComputeKernelProjectionsAutocorrelation(h: np.ndarray, AngleSet: np.ndarray, Mh: int, r: int = 4) -> np.ndarray:
    win_len = r * Mh + 1
    Rh = np.zeros((len(AngleSet), win_len), dtype=np.float32)

    for i, angle in enumerate(AngleSet):
        q = _shear_projection_kernel(h, angle)

        ac = _autocorr_1d(q, Mh, r=r)  
        L = ac.shape[0]

        if L == win_len:
            Rh[i, :] = ac
        elif L > win_len:
            c_src = L // 2
            half = win_len // 2
            Rh[i, :] = ac[c_src - half : c_src + half + 1]
        else:
            pad = np.zeros(win_len, dtype=np.float32)
            c_src = L // 2
            c_dst = win_len // 2
            start = c_dst - c_src
            end = start + L
            pad[start:end] = ac
            Rh[i, :] = pad

    return Rh


def ReestimateSupport(h: np.ndarray, AngleSet: np.ndarray, Mh: int, threshold_factor: float = 0.05, r: int = 4) -> np.ndarray:

    Rh = ComputeKernelProjectionsAutocorrelation(h, AngleSet, Mh, r=r)
    n_angles, win_len = Rh.shape
    center = win_len // 2

    S = np.zeros(n_angles, dtype=np.int32)

    for i in range(n_angles):
        ac = Rh[i, :]

        ac_pos = ac[center + 1:]
        if ac_pos.size == 0:
            S[i] = 0
            continue

        max_val = ac_pos.max()
        if max_val <= 0:
            S[i] = 0
            continue

        thresh = threshold_factor * max_val

        valid = np.where(ac_pos > thresh)[0]

        if valid.size == 0:
            S[i] = 0
        else:
            S[i] = int(valid[-1] + 1)

    return S

def InitialSupportEstimation(R: np.ndarray,  Mh: int, r: int = 4, kappa: float = 2/70) -> np.ndarray:
    n_angles, win_len = R.shape
    center = win_len // 2

    s_min = np.zeros(n_angles, dtype=np.int32)
    for i in range(n_angles):
        ac = R[i, :]
        ac_pos = ac[center+1:]
        k_rel = np.argmin(ac_pos) + 1  
        s_min[i] = k_rel

    S = np.full(n_angles, r * Mh, dtype=np.int32)

    for i in range(n_angles):
        if s_min[i] < S[i]:
            S[i] = s_min[i]
            for j in range(n_angles):
                S[j] = min(S[j], s_min[i] + kappa * abs(i - j))

    return S



def EstimatePowerSpectrum(R: np.ndarray, EstimatedSupport: np.ndarray, AngleSet: np.ndarray, r: int = 4) -> np.ndarray:
    n_angles, win_len = R.shape
    center = win_len // 2
    Mh = (win_len - 1) // r

    Rh = np.zeros_like(R, dtype=np.float32)
    for i in range(n_angles):
        ac = R[i].astype(np.float32)
        s = int(EstimatedSupport[i])
        if s <= 0 or s >= center:
            continue

        mu = ac[center + s]
        k_min = center - s
        k_max = center + s
        ac_trunc = ac[k_min:k_max + 1] - mu
        ac_trunc = np.maximum(ac_trunc, 0.0)
        Rh[i, k_min:k_max + 1] = ac_trunc

        ssum = ac_trunc.sum()
        if ssum > 0:
            Rh[i] /= ssum

    med_win = max(3, int(2 * np.sqrt(n_angles)) | 1)
    Rh_filt = scipy.ndimage.median_filter(Rh, size=(med_win, 1), mode='nearest')

    H2 = np.zeros((Mh, Mh), dtype=np.float32)
    cx = cy = Mh // 2

    assert len(AngleSet) == n_angles, "AngleSet et R doivent avoir la même taille"

    for i, theta in enumerate(AngleSet):
        ac_theta = Rh_filt[i]

        tmp = np.zeros_like(ac_theta)
        tmp[:center + 1] = ac_theta[center:]
        tmp[-center:] = ac_theta[:center]

        spec_1d = np.fft.fft(tmp)
        mag_1d = np.abs(spec_1d).astype(np.float32)

        max_radial = win_len // 2 + 1
        Nr = min(max_radial, cx + 1)
        if Nr <= 1:
            continue

        for n in range(Nr):
            rho = n * (cx / (Nr - 1)) if Nr > 1 else 0.0
            x = cx + rho * np.cos(theta)
            y = cy + rho * np.sin(theta)
            ix = int(round(x))
            iy = int(round(y))
            if 0 <= ix < Mh and 0 <= iy < Mh:
                val = mag_1d[n]
                if H2[iy, ix] == 0:
                    H2[iy, ix] = val
                else:
                    H2[iy, ix] = 0.5 * (H2[iy, ix] + val)

    ssum = H2.sum()
    if ssum > 0:
        H2 /= ssum

    return H2

def SinglePhaseRetrieval(H: np.ndarray, Mh: int, alph: float = 0.95, beta0: float = 0.75, Ninner: int = 300) -> np.ndarray:

    H = np.asarray(H, dtype=np.float32)
    P, Q = H.shape
    assert P == Q, "Spectrum H must be square."

    H_mag = np.sqrt(np.maximum(H, 0.0))

    rand_spatial = np.random.randn(P, P).astype(np.float32)
    G0 = np.fft.fft2(rand_spatial)
    phase = np.angle(G0)

    G = H_mag * np.exp(1j * phase)
    g = np.fft.ifft2(G).real

    support_mask = np.zeros_like(g, dtype=bool)
    support_mask[:Mh, :Mh] = True

    alpha = alph

    g0 = g.copy()
    for m in range(Ninner):
        beta = beta0 + (1.0 - beta0) * (1.0 - np.exp(-(m / 7.0) ** 3))

        G = np.fft.fft2(g)
        G_mag = np.abs(G)
        phase = np.angle(G)

        mixed_mag = alpha * H_mag + (1.0 - alpha) * G_mag
        G_new = mixed_mag * np.exp(1j * phase)
        g0 = np.fft.ifft2(G_new).real

        omega = (2.0 * g0 < g) | (~support_mask)

        g_updated = g0.copy()
        g_updated[omega] = beta * g[omega] + (1.0 - 2.0 * beta) * g0[omega]
        g = g_updated

    h = g0[:Mh, :Mh].copy()

    h = np.maximum(h, 0.0)

    ssum = h.sum()
    if ssum > 0:
        h /= ssum

    thresh = 1.0 / 255.0
    h[h < thresh] = 0.0
    ssum = h.sum()
    if ssum > 0:
        h /= ssum

    return h.astype(np.float32)

def PhaseRetrieval(v: np.ndarray, H: np.ndarray, Mh: int, Ntries: int = 10) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    H_img, W_img = v.shape
    P = _extract_patch(v)

    best_score = np.inf
    best_h = None

    for _ in range(Ntries):
        h_candidate = SinglePhaseRetrieval(H, Mh)
        h_reflect = np.flipud(np.fliplr(h_candidate))

        s1 = _score_kernel(h_candidate, P)
        s2 = _score_kernel(h_reflect, P)

        if s2 < s1:
            s1 = s2
            h_candidate = h_reflect

        if s1 < best_score or best_h is None:
            best_score = s1
            best_h = h_candidate


    h_centered = center_kernel(best_h)
    return h_centered



def blur_kernel_estimation(v: np.ndarray, p=25, alph: float = 1, Nouter: int = 3, verbose: bool = False) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    if v.ndim == 3:
        v = v.mean(axis=2)

    Mh = p
    A = ComputeProjectionAngleSet(Mh)
    if verbose:
        print(f"Number of projection angles: {len(A)} \n First angles (radians): {A[:5]}")


    # R = ComputeProjectionsAutocorrelation(v, A, Mh, alph=alph, r=4, verbose=verbose)
    R = compute_R(v, A, p)
    if verbose:
        print("Computed projections' autocorrelations : len(R) =", R.shape)
        show_autocorr(R)
        plt.pause(10)
        plt.close()


    S = InitialSupportEstimation(R, Mh, r=4, kappa=2/70)
    if verbose:
        print("Initial support estimation done : len(S) =", S.shape)
        plt.plot(S)
        plt.title("Estimated support sizes per angle")
        plt.show()
        plt.pause(5)
        plt.close()

    h = np.zeros((Mh, Mh), dtype=np.float32)

    for i in range(Nouter):
        if verbose:
            print(f"--- Outer iteration {i+1}/{Nouter} ---")

        H = EstimatePowerSpectrum(R, S, A, r=4)

        if verbose: 
            print("Estimated power spectrum of the blur kernel : len(H), sum(H) =", H.shape, H.sum())
            plt.imshow(H, cmap='hot')
            plt.title(f"Estimated Power Spectrum at iteration {i+1}")
            plt.show()
            plt.pause(5)
            plt.close()

        h = PhaseRetrieval(v, H, Mh)

        if verbose:
            print("Phase retrieval done : len(h), sum(h) =", h.shape, h.sum())
            plt.plot(h)
            plt.title(f"Estimated Kernel at iteration {i+1}")
            plt.show()
            plt.pause(5)
            plt.close()

        S = ReestimateSupport(h, A, Mh, threshold_factor=0.05)

    print(f"Final Kernel Estimated, h.size: {h.shape}")
    return h



if __name__ == "__main__":
    Mh = 25
    A = ComputeProjectionAngleSet(Mh)
    plt.plot(A)