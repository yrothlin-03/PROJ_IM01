import numpy as np
from tv_deconv import tv_deconv
import scipy


def _derivatives(u: np.ndarray):
    """Compute horizontal and vertical discrete derivatives vx, vy using
    the 1D filter d given in Algorithm 2 of the IPOL article
    (Anger–Facciolo–Delbracio, 2018) 
    d = [3, -32, 168, -672, 0, 672, -168, 32, -3] / 840.

    Returns
    -------
    vx, vy : np.ndarray
        Horizontal and vertical derivatives (same shape as input).
    """
    d = np.array([3, -32, 168, -672, 0, 672, -168, 32, -3], dtype=np.float64) / 840.0
    k = d.size // 2

    # Horizontal derivative: convolve each row with d (replicate boundary)
    H, W = u.shape[:2]
    pad_width = ((0, 0), (k, k)) if u.ndim == 2 else ((0, 0), (k, k), (0, 0))
    up = np.pad(u, pad_width, mode='edge')
    if u.ndim == 2:
        vx = np.apply_along_axis(lambda r: np.convolve(r, d, mode='valid'), axis=1, arr=up)
    else:
        # If a channel dimension exists, process per channel
        vx = np.stack([
            np.apply_along_axis(lambda r: np.convolve(r, d, mode='valid'), axis=1, arr=up[..., c])
            for c in range(u.shape[2])
        ], axis=2)

    # Vertical derivative: convolve each column with d^T (replicate boundary)
    pad_width = ((k, k), (0, 0)) if u.ndim == 2 else ((k, k), (0, 0), (0, 0))
    up = np.pad(u, pad_width, mode='edge')
    if u.ndim == 2:
        vy = np.apply_along_axis(lambda r: np.convolve(r, d, mode='valid'), axis=0, arr=up)
    else:
        vy = np.stack([
            np.apply_along_axis(lambda r: np.convolve(r, d, mode='valid'), axis=0, arr=up[..., c])
            for c in range(u.shape[2])
        ], axis=2)

    return vx.astype(np.float64), vy.astype(np.float64)

def _shear_project(v: np.ndarray, theta):
    """Shear projection with nearest-neighbor accumulation (Algorithm 2).

    For |theta| <= pi/4:  horizontal shear  offset = x_c + y_c * tan(theta)
    Otherwise:            vertical shear    offset = y_c + x_c / tan(theta)

    Coordinates are centered at the image center before shearing and the result
    is accumulated into a 1D array q indexed by the rounded offsets. The length
    of q is exactly max_off - min_off + 1 so that the projection is compact.
    """
    if v.ndim != 2:
        raise ValueError("_shear_project expects a 2D array")

    H, W = v.shape
    cx = (W - 1) / 2.0
    cy = (H - 1) / 2.0

    yy, xx = np.meshgrid(np.arange(H, dtype=np.float64), np.arange(W, dtype=np.float64), indexing='ij')
    x_c = xx - cx
    y_c = yy - cy

    # Choose shear according to angle
    if -np.pi/4.0 <= theta <= np.pi/4.0:
        # Horizontal shear
        offsets = x_c + y_c * np.tan(theta)
    else:
        # Vertical shear
        # avoid division by zero when theta is 0 or +/- pi/2 is handled by branch
        offsets = y_c + x_c / np.tan(theta)

    # Round to nearest integer sample position (nearest-neighbor accumulation)
    o_int = np.rint(offsets).astype(np.int64)
    min_off = int(o_int.min())
    max_off = int(o_int.max())
    q = np.zeros(max_off - min_off + 1, dtype=np.float64)

    # Accumulate values at their rounded offsets
    flat_idx = (o_int - min_off).ravel()
    np.add.at(q, flat_idx, v.ravel())

    return q

def _autocorr1D(x: np.ndarray):
    """Return the discrete 1D autocorrelation of x (zero-lag centered).

    Implemented as the full correlation of the signal with itself.
    Output length is 2*N-1 and the zero-lag corresponds to index N-1.
    """
    x = np.asarray(x, dtype=np.float64)
    r = np.correlate(x, x, mode='full')  # autocorrelation
    return r

def _median_filter(x: np.ndarray, angles: np.ndarray):
    """Apply a median filter across the angle axis (per Algorithm 4 note).

    Parameters
    ----------
    x : np.ndarray
        2D array of shape (num_angles, L) containing per-angle signals
        (e.g., autocorrelations). This function returns a filtered copy.
    angles : np.ndarray
        Array of angles A used to set the filter window size as
        k = int(2*sqrt(|A|)). The window is made odd and at least 1.
    """
    if x.ndim != 2:
        raise ValueError("_median_filter expects a 2D array of shape (num_angles, L)")
    nA = int(len(angles))
    k = int(2 * np.sqrt(max(nA, 1)))
    if k % 2 == 0:
        k += 1
    k = max(k, 1)

    # Sliding-window median along axis=0 (angles) for each column independently
    num_angles, L = x.shape
    out = np.empty_like(x, dtype=np.float64)
    half = k // 2
    for i in range(num_angles):
        start = max(0, i - half)
        end = min(num_angles, i + half + 1)
        # median over the angle window for all columns at once
        out[i, :] = np.median(x[start:end, :], axis=0)
    return out



def ComputeProjectionAngleSet(p: int):
    r = 4
    Mh = p
    lim = r * Mh // 2
    tangents = set()
    for i in range(-lim, lim + 1):
        for j in range(-lim, lim + 1):
            if i == 0 and j == 0:
                continue
            if i == 0:
                theta = np.pi / 2.0
            else:
                theta = np.arctan2(j, i)
            if theta <= -np.pi/2:
                theta += np.pi
            if theta > np.pi/2:
                theta -= np.pi
            tangents.add(float(theta))
    A = np.array(sorted(list(tangents), reverse=True), dtype=np.float64)
    return A



def ComputeProjectionsAutocorrelation(v: np.ndarray, angles: np.ndarray, p: int, alph: float):
    """Compute per-angle autocorrelation of sheared gradient projections.

    For each angle θ, we form the 1D projection q_θ by shearing the horizontal
    and vertical derivatives and summing them, then compute the 1D
    autocorrelation r_θ. We keep only the central window of length (2*p-1)
    (lags from -(p-1) to +(p-1)) and apply an exponential taper
    exp(-alph * |lag|) to de-emphasize far lags.

    Parameters
    ----------
    v : np.ndarray
        Input 2D image.
    angles : np.ndarray
        Angle list (in radians).
    p : int
        Target kernel size; controls the autocorrelation window (2*p-1).
    alph : float
        Damping parameter for the exponential taper over lags.

    Returns
    -------
    R : np.ndarray
        Array of shape (len(angles), 2*p-1) with per-angle autocorrelations.
    """
    if v.ndim != 2:
        raise ValueError("ComputeProjectionsAutocorrelation expects a 2D image array")

    # Derivatives
    vx, vy = _derivatives(v)

    L = 2 * p - 1
    R_list = []

    for theta in angles:
        # Shear-projection of derivatives and accumulation along the shear axis
        qx = _shear_project(vx, theta)
        qy = _shear_project(vy, theta)
        q = qx + qy

        # Autocorrelation (full), zero-lag at index len(q)-1
        r_full = _autocorr1D(q)
        center = len(q) - 1

        # Extract central window of size L = 2*p - 1
        half = p - 1
        start = max(0, center - half)
        end = min(len(r_full), center + half + 1)
        r = np.zeros(L, dtype=np.float64)
        # Place the available slice into the centered window
        r[(half - (center - start)):(half + (end - center))] = r_full[start:end]

        # Exponential taper by lag distance
        lags = np.arange(-half, half + 1, dtype=np.float64)
        if alph is not None and alph > 0:
            w = np.exp(-alph * np.abs(lags))
            r *= w

        R_list.append(r)

    R = np.vstack(R_list) if len(R_list) else np.zeros((0, L), dtype=np.float64)
    return R

    

def InitialSupportEstimation(R: np.ndarray, angles: np.ndarray):
    """Estimate initial per-angle support s_θ from autocorrelations R.

    Implements Algorithm 3 from the IPOL article (Anger–Facciolo–Delbracio, 2018):
    - For each angle, set s'_θ as the positive-index location of the minimum of the
      autocorrelation R(P_θ(D_θ v)).
    - Initialize s_θ to the maximum admissible lag (here p-1 when R has length 2p-1).
    - Enforce a Lipschitz continuity constraint across angles:
          s_{θ_j} = min(s_{θ_j}, s'_{θ_i} + κ * |i - j|)
      with κ = 2/70 and i, j the indices in the provided angles array.

    Parameters
    ----------
    R : np.ndarray
        Autocorrelations per angle, shape (num_angles, L). Zero-lag is at index L//2.
    angles : np.ndarray
        Angles array (only its length/order matters here).

    Returns
    -------
    s : np.ndarray
        Integer support per angle (num_angles,), positive lag indices.
    """
    if R.ndim != 2:
        raise ValueError("R must be a 2D array of shape (num_angles, L)")
    num_angles, L = R.shape
    center = L // 2

    # 1) Raw minima on positive lags (k > 0)
    if L - (center + 1) <= 0:
        raise ValueError("R rows are too short to contain positive lags")
    pos = R[:, center + 1 :]
    # argmin over the positive side; add 1 to convert back to absolute index offset from center
    s_prime = np.argmin(pos, axis=1) + 1  # distances in samples (k > 0)

    # 2) Initialize s with the maximum allowed positive lag (here Lpos_max = L-1-center)
    Lpos_max = L - 1 - center
    s = np.full(num_angles, Lpos_max, dtype=np.int64)

    # 3) Lipschitz continuity propagation with κ = 2/70
    kappa = 2.0 / 70.0
    for i in range(num_angles):
        if s_prime[i] < s[i]:
            s[i] = s_prime[i]
            # propagate to all other angles
            di = np.abs(np.arange(num_angles) - i)
            bound = np.floor(s_prime[i] + kappa * di).astype(np.int64)
            bound = np.clip(bound, 1, Lpos_max)  # ensure valid positive-lag indices
            s = np.minimum(s, bound)

    return s

def EstimatePowerSpectrum(R: np.ndarray, S: np.ndarray, angles: np.ndarray):
    """Estimate a 2D power spectrum from per-angle autocorrelations.

    Steps (following the spirit of Algorithm 4):
      1) Enforce per-angle support using S (zero out |lag| > S[θ]).
      2) Median-filter across angles to stabilize noisy lines.
      3) For each angle θ, compute the 1D power spectrum via FFT of the
         autocorrelation (Wiener–Khinchin). Ensure nonnegativity and center.
      4) Backproject these polar spectra using the provided angle set onto a square Cartesian grid.
      5) Normalize the resulting 2D spectrum.

    Parameters
    ----------
    R : np.ndarray
        Array of autocorrelations per angle, shape (num_angles, L),
        with zero-lag at index L//2 in each row.
    S : np.ndarray
        Integer support per angle (num_angles,), specifying the maximum
        positive lag to keep for each angle. Entries outside ±S[θ] are zeroed.
    angles : np.ndarray
        Angles used to compute R (in radians), same order/length as R rows.

    Returns
    -------
    P : np.ndarray
        Estimated 2D power spectrum on an L×L Cartesian grid.
    """
    if R.ndim != 2:
        raise ValueError("R must be 2D (num_angles, L)")
    num_angles, L = R.shape
    if S.shape[0] != num_angles:
        raise ValueError("S must have length equal to number of angles")

    center = L // 2

    # 1) Apply per-angle support (keep lags in [-S[i], +S[i]])
    R_supported = np.zeros_like(R, dtype=np.float64)
    for i in range(num_angles):
        s = int(S[i])
        left = max(0, center - s)
        right = min(L, center + s + 1)
        R_supported[i, left:right] = R[i, left:right]

    # 2) Median filter across angles to reduce outliers
    R_filt = _median_filter(R_supported, np.arange(num_angles, dtype=np.float64))

    # 3) 1D power spectrum per angle (Wiener–Khinchin)
    # Move zero-lag to index 0 -> FFT -> keep real part -> ensure nonnegative -> center (fftshift)
    polar_ps = np.zeros_like(R_filt, dtype=np.float64)
    for i in range(num_angles):
        r = R_filt[i]
        # Normalize by zero-lag to avoid arbitrary scale differences (if nonzero)
        if r[center] != 0:
            r = r / float(r[center])
        r0 = np.fft.ifftshift(r)
        ps = np.real(np.fft.fft(r0))
        ps = np.maximum(ps, 0.0)  # numerical floor
        polar_ps[i] = np.fft.fftshift(ps)

    # 4) Backproject polar spectra onto Cartesian grid (nearest-neighbor)
    P = np.zeros((L, L), dtype=np.float64)
    cx = cy = (L - 1) / 2.0

    # Precompute for angle nearest-neighbor lookup
    A = np.asarray(angles, dtype=np.float64)

    # Angle list monotonic assumption; build fast mapping by binning
    for y in range(L):
        dy = y - cy
        for x in range(L):
            dx = x - cx
            rho = np.hypot(dx, dy)
            if rho > center:  # outside maximum representable radius
                continue
            theta = np.arctan2(dy, dx)
            # Map angle to principal domain [-pi/2, pi/2] to match our projections
            if theta <= -np.pi/2:
                theta += np.pi
            if theta > np.pi/2:
                theta -= np.pi
            # Nearest angle index
            i = int(np.argmin(np.abs(A - theta)))
            k = int(np.rint(rho))
            P[y, x] = polar_ps[i, (center - k) + center]  # index centered spectrum by radius

    # 5) Normalize spectrum (scale-invariant)
    m = P.max()
    if m > 0:
        P = P / m

    return P

def SinglePhaseRetrieval(P: np.ndarray, p:int, Ninner: int = 300):
    """Single run (Algorithm 5): RAAR-like phase retrieval with attraction to |H|.

    Enforces: Fourier magnitude (attraction with α), support p×p, positivity, normalization,
    and simple thresholding 1/255, as in the IPOL article.
    """
    if P.ndim != 2:
        raise ValueError("P must be 2D")
    Ly, Lx = P.shape
    if Ly != Lx:
        raise ValueError("P must be square")
    L = Lx
    if p > L:
        raise ValueError("p must be <= size of P")

    # Parameters from Algorithm 5
    alpha = 0.95
    beta0 = 0.75

    # Target magnitude (DC at [0,0])
    A = np.sqrt(np.maximum(P.astype(np.float64), 0.0))
    A0 = np.fft.ifftshift(A)

    # Centered p×p support in an L×L grid
    c = (L - 1) // 2
    half = p // 2
    y = np.arange(L)
    x = np.arange(L)
    Y, X = np.meshgrid(y, x, indexing='ij')
    support = (np.abs(Y - c) <= half) & (np.abs(X - c) <= half)

    # Hermitian-symmetric random phase init (via random real spatial g)
    rng = np.random.default_rng()
    g = rng.random((L, L))
    G = np.fft.fft2(g)

    for m in range(int(Ninner)):
        # Update β schedule
        beta = beta0 + (1.0 - beta0) * (1.0 - np.exp(- (m / 7.0) ** 3))

        # Current magnitude and phase
        G = np.fft.fft2(g)
        absG = np.abs(G)
        angG = np.angle(G)

        # Enforce spectrum magnitude (attraction to |H|)
        mixed_mag = alpha * A0 + (1.0 - alpha) * absG
        Gp = mixed_mag * np.exp(1j * angG)
        g_prime = np.real(np.fft.ifft2(Gp))

        # Define invalid set Ω
        Omega = (2.0 * g_prime < g) | (~support)

        # RAAR update
        g_new = g_prime.copy()
        g_new[Omega] = beta * g[Omega] + (1.0 - 2.0 * beta) * g_prime[Omega]
        g = g_new

        # Positivity + normalization inside support (soft enforcement each iter)
        g[~support] = 0.0
        g[g < 0.0] = 0.0
        ssum = g.sum()
        if ssum > 0:
            g /= ssum

    # Final kernel = constrained crop from last estimate
    top = c - half
    bot = c + half + 1
    h = g[top:bot, top:bot].copy()

    # Thresholding at 1/255 and renormalize (as in Alg. 5)
    if h.size:
        h[h < (1.0/255.0)] = 0.0
        ssum = h.sum()
        if ssum > 0:
            h /= ssum
        else:
            h[p//2, p//2] = 1.0
    return h

def RetrievePhase(P: np.ndarray, p: int, v: np.ndarray, Ntries: int = 30):
    """Algorithm 6: multi-start phase retrieval with deconvolution-based selection.

    - Extract a high-variance 150×150 patch from v (among 10 random windows).
    - Run SinglePhaseRetrieval Ntries times; also evaluate the mirrored kernel.
    - Score each candidate by the ℓ1/ℓ2 ratio of gradient magnitudes after TV deconvolution.
    - Return the kernel with the lowest score. Kernels are centered by centroid before use.
    """
    # 1) Extract a high-variance 150×150 patch (10 random tries)
    H, W = v.shape[:2]
    win = 150
    if H < win or W < win:
        Pimg = v.copy()
    else:
        rng = np.random.default_rng(123)
        best_var = -np.inf
        best_patch = None
        for _ in range(10):
            y0 = rng.integers(0, H - win + 1)
            x0 = rng.integers(0, W - win + 1)
            patch = v[y0:y0+win, x0:x0+win]
            var = float(np.var(patch))
            if var > best_var:
                best_var = var
                best_patch = patch
        Pimg = best_patch if best_patch is not None else v.copy()

    def center_kernel(hk: np.ndarray) -> np.ndarray:
        hk = np.maximum(hk, 0.0)
        s = hk.sum()
        if s <= 0:
            hk[hk.shape[0]//2, hk.shape[1]//2] = 1.0
            s = 1.0
        hk = hk / s
        # centroid shift to center
        yy, xx = np.meshgrid(np.arange(hk.shape[0]), np.arange(hk.shape[1]), indexing='ij')
        cy = float((yy * hk).sum()) / s
        cx = float((xx * hk).sum()) / s
        sy = int(round(hk.shape[0]//2 - cy))
        sx = int(round(hk.shape[1]//2 - cx))
        return np.roll(np.roll(hk, sy, axis=0), sx, axis=1)

    def score_kernel(hk: np.ndarray) -> float:
        # TV deconvolution of the patch followed by l1/l2 gradient score
        try:
            d = tv_deconv(Pimg.astype(np.float64), hk.astype(np.float64))
            # Some tv_deconv implementations return (result, info). Keep the image.
            if isinstance(d, (tuple, list)):
                d = d[0]
        except Exception:
            # Fallback: simple no-deconv score on the patch
            d = Pimg.astype(np.float64)
        d = np.asarray(d, dtype=np.float64)
        if d.ndim == 3:
            # If color, convert to luminance
            d = 0.2989 * d[..., 0] + 0.5870 * d[..., 1] + 0.1140 * d[..., 2]
        # finite differences with replicated boundary
        gx = np.diff(d, axis=1, append=d[:, -1:])
        gy = np.diff(d, axis=0, append=d[-1:, :])
        mag = np.abs(gx) + np.abs(gy)
        num = float(np.sum(mag))
        den = float(np.sqrt(np.sum(mag**2)) + 1e-12)
        return num / den

    best_score = np.inf
    best_h = None
    for _ in range(int(max(1, Ntries))):
        h0 = SinglePhaseRetrieval(P, p)
        h0 = center_kernel(h0)
        s0 = score_kernel(h0)
        if s0 < best_score:
            best_score = s0
            best_h = h0
        # evaluate mirrored kernel
        hm = center_kernel(np.flip(np.flip(h0, axis=0), axis=1))
        sm = score_kernel(hm)
        if sm < best_score:
            best_score = sm
            best_h = hm

    if best_h is None:
        best_h = np.zeros((p, p), dtype=np.float64)
        best_h[p//2, p//2] = 1.0
    return best_h

def ReestimateSupport(h: np.ndarray, angles: np.ndarray):
    """Re-estimate per-angle support S from the current kernel h.

    Uses a binary support mask from h (thresholded at 1% of max) and measures,
    for each angle, the furthest occupied offset along the corresponding shear
    axis. This mirrors the geometry used in the autocorrelation projections.

    Parameters
    ----------
    h : np.ndarray
        Current kernel estimate, shape (p, p), nonnegative.
    angles : np.ndarray
        Sequence of projection angles (in radians).

    Returns
    -------
    S : np.ndarray
        Integer support per angle (num_angles,), with entries in [1, p-1].
    """
    if h.ndim != 2:
        raise ValueError("h must be a 2D array (p, p)")
    H, W = h.shape
    if H != W:
        raise ValueError("h must be square (p×p)")
    p = H

    # Build binary support mask (robust to small noise)
    h = np.asarray(h, dtype=np.float64)
    hmax = h.max()
    tau = 0.01 * hmax if hmax > 0 else 0.0  # 1% of max as in practice
    mask = h > tau
    if not np.any(mask):
        # Fall back to minimal support
        return np.ones(len(angles), dtype=np.int64)

    # Centered coordinates
    cx = (W - 1) / 2.0
    cy = (H - 1) / 2.0
    yy, xx = np.meshgrid(np.arange(H, dtype=np.float64), np.arange(W, dtype=np.float64), indexing='ij')
    x_c = xx - cx
    y_c = yy - cy

    S = np.empty(len(angles), dtype=np.int64)
    max_allow = p - 1  # maximum positive lag representable in R of length 2p-1

    for k, theta in enumerate(angles):
        if -np.pi/4.0 <= theta <= np.pi/4.0:
            offsets = x_c + y_c * np.tan(theta)
        else:
            offsets = y_c + x_c / np.tan(theta)
        o_int = np.rint(offsets[mask]).astype(np.int64)
        if o_int.size == 0:
            s = 1
        else:
            s = int(np.max(np.abs(o_int)))
            s = max(1, min(s, max_allow))
        S[k] = s

    return S

def blur_kernel_estimation(v: np.ndarray, K: np.ndarray, alph: float = 2.1, Nouter: int = 3):
    if isinstance(K, int):
        p = K
    else:
        p = K.shape[0]

    angles = ComputeProjectionAngleSet(p)

    R = ComputeProjectionsAutocorrelation(v, angles, p, alph)

    S = InitialSupportEstimation(R, angles)

    for it in range(Nouter):
        P = EstimatePowerSpectrum(R, S, angles)

        h = RetrievePhase(P, p, v)
 
        S = ReestimateSupport(h, angles)

    return h

