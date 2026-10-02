"""Shared helpers for astrotilt validation: synthetic star rendering, the
initial (pre-fix) SEP extractor, and a per-star port of the original
(pre-SEP) astrotilt algorithm.

The original algorithm is loaded from a checkout of
https://github.com/jfkingiii/astrotilt given by $ASTROTILT_ORIG
(default /tmp/astrotilt-orig); only ``old_extract`` needs it.
"""

import importlib.util
import os

import numpy as np
import sep
from astropy.stats import mad_std

from astrotilt.stars import eccentricity_from_axes, extract_stars  # noqa: F401  (fixed version)

_orig_stars = None


def _load_orig_stars():
    global _orig_stars
    if _orig_stars is None:
        root = os.environ.get("ASTROTILT_ORIG", "/tmp/astrotilt-orig")
        path = os.path.join(root, "src", "astrotilt", "stars.py")
        if not os.path.exists(path):
            raise RuntimeError(
                f"original astrotilt not found at {root}; run "
                "'git clone https://github.com/jfkingiii/astrotilt /tmp/astrotilt-orig' "
                "or set $ASTROTILT_ORIG"
            )
        spec = importlib.util.spec_from_file_location("orig_stars", path)
        _orig_stars = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_orig_stars)
    return _orig_stars


def sep_initial_extract(image, nsigma=5.0, min_pixels=5, max_pixels=1000):
    """The SEP extractor as first reviewed (v0.2.0 before fixes): SEP's own
    a/b (measured on the filtered image), no flag, S/N or saturation cuts."""
    data = np.ascontiguousarray(image, dtype=np.float32)
    bkg = sep.Background(data)
    o = sep.extract(data - bkg.back(dtype=np.float32), float(nsigma),
                    err=bkg.rms(dtype=np.float32), minarea=int(min_pixels))
    ecc = eccentricity_from_axes(o["a"], o["b"])
    keep = np.isfinite(ecc) & (o["npix"] >= min_pixels) & (o["npix"] <= max_pixels)
    out = np.zeros(int(keep.sum()), dtype=[(n, "f8") for n in
                                            ("x", "y", "npix", "a", "b", "theta", "flag", "eccentricity")])
    for n in ("x", "y", "npix", "a", "b", "theta", "flag"):
        out[n] = o[n][keep]
    out["eccentricity"] = ecc[keep]
    return out


OS = 5  # oversampling factor for pixel-integrated rendering


def psf_stamp(profile, fwhm, q, theta, dx, dy, half, beta=3.0):
    """Unit-flux pixel-integrated PSF on a (2*half+1)^2 stamp.

    profile: 'gauss' or 'moffat'. fwhm is along the MAJOR axis; minor axis = q*major.
    theta: major-axis angle, radians CCW from +x (array column) toward +y (array row).
    dx, dy: subpixel centroid offsets relative to stamp centre.
    """
    n = 2 * half + 1
    s = (np.arange(n * OS) + 0.5) / OS - 0.5 - half
    yy, xx = np.meshgrid(s - dy, s - dx, indexing="ij")
    c, si = np.cos(theta), np.sin(theta)
    u = xx * c + yy * si       # along major axis
    v = -xx * si + yy * c      # along minor axis
    if profile == "gauss":
        sig = fwhm / 2.3548
        r2 = (u / sig) ** 2 + (v / (q * sig)) ** 2
        img = np.exp(-0.5 * r2)
    else:
        alpha = fwhm / (2.0 * np.sqrt(2 ** (1.0 / beta) - 1.0))
        r2 = (u / alpha) ** 2 + (v / (q * alpha)) ** 2
        img = (1.0 + r2) ** (-beta)
    img = img.reshape(n, OS, n, OS).sum(axis=(1, 3))
    return img / img.sum()


def snr_to_flux(snr, fwhm, q, sky_sigma):
    """Background-limited optimal (matched-filter) S/N for a Gaussian:
    SNR = F / (sigma_sky * sqrt(4*pi*sx*sy))."""
    sx = fwhm / 2.3548
    return snr * sky_sigma * np.sqrt(4.0 * np.pi * sx * (q * sx))


def make_field(rng, n_side, spacing, fwhm, ecc, snr, profile="gauss",
               sky=1000.0, sky_sigma=10.0, theta=None, gradient=None,
               offset=(0, 0)):
    """Grid of isolated stars with random subpixel centroids.

    Returns (image, truth) where truth is a structured array of x, y, theta.
    ecc is the TRUE eccentricity sqrt(1-q^2) of the PSF isophotes.
    """
    q = np.sqrt(1.0 - ecc ** 2)
    H = W = n_side * spacing
    img = np.full((H, W), sky, dtype=np.float64)
    if gradient is not None:
        img += gradient(H, W)
    half = int(np.ceil(4 * fwhm)) + 3
    truth = []
    for i in range(n_side):
        for j in range(n_side):
            cx = j * spacing + spacing // 2 + offset[0]
            cy = i * spacing + spacing // 2 + offset[1]
            dx, dy = rng.uniform(-0.5, 0.5, 2)
            th = rng.uniform(-np.pi / 2, np.pi / 2) if theta is None else theta
            flux = snr_to_flux(snr, fwhm, q, sky_sigma)
            st = flux * psf_stamp(profile, fwhm, q, th, dx, dy, half)
            y0, x0 = cy - half, cx - half
            ys, xs = slice(max(y0, 0), min(y0 + 2 * half + 1, H)), slice(max(x0, 0), min(x0 + 2 * half + 1, W))
            img[ys, xs] += st[ys.start - y0: ys.stop - y0, xs.start - x0: xs.stop - x0]
            truth.append((cx + dx, cy + dy, th))
    img += rng.normal(0, sky_sigma, img.shape)
    truth = np.array(truth, dtype=[("x", "f8"), ("y", "f8"), ("theta", "f8")])
    return img.astype(np.float32), truth


def old_extract(image, nsigma=5.0, min_pixels=5, max_pixels=1000, grid=3):
    """Original astrotilt algorithm (per 3x3 cell, raw-image moments), but
    returning per-star (x, y, ecc) instead of only the cell median."""
    orig = _load_orig_stars()
    H, W = image.shape
    ch, cw = H // grid, W // grid
    out = []
    for r in range(grid):
        for c in range(grid):
            r0, r1 = r * ch, (H if r == grid - 1 else (r + 1) * ch)
            c0, c1 = c * cw, (W if c == grid - 1 else (c + 1) * cw)
            cell = image[r0:r1, c0:c1]
            med, sd = np.median(cell), mad_std(cell)
            rows, cols = np.where(cell > med + nsigma * sd)
            if len(rows) == 0:
                continue
            vals = cell[rows, cols]
            labels = orig.union_find_labels(rows, cols, cell.shape)
            for lbl in np.unique(labels):
                m = labels == lbl
                if not (min_pixels <= m.sum() <= max_pixels):
                    continue
                e = orig.eccentricity_from_moments(rows[m], cols[m], vals[m])
                if np.isnan(e):
                    continue
                w = vals[m].astype(float)
                out.append((c0 + np.dot(cols[m], w) / w.sum(), r0 + np.dot(rows[m], w) / w.sum(), e))
    return np.array(out, dtype=[("x", "f8"), ("y", "f8"), ("eccentricity", "f8")])


def match(det, truth, tol=2.0):
    """Index of nearest truth for each detection (or -1 if > tol px)."""
    if len(det["x"]) == 0:
        return np.empty(0, int)
    d2 = (det["x"][:, None] - truth["x"][None]) ** 2 + (det["y"][:, None] - truth["y"][None]) ** 2
    idx = d2.argmin(axis=1)
    idx[d2[np.arange(len(det["x"])), idx] > tol ** 2] = -1
    return idx


__all__ = ["extract_stars", "sep_initial_extract", "old_extract", "make_field", "psf_stamp", "match", "snr_to_flux"]
