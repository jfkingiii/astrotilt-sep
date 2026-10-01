"""
stars.py — SEP-based star detection and morphology.

SEP (the Python implementation of Source Extractor algorithms) handles
spatially varying background estimation, source detection, filtering,
deblending, and second-moment ellipse measurements.
"""

import numpy as np
import sep


def _as_sep_image(image):
    """Return a native-endian, C-contiguous float32 2-D array for SEP."""
    data = np.asarray(image, dtype=np.float32)
    if data.ndim != 2:
        raise ValueError(f"Expected a 2-D image, got shape {data.shape}")
    return np.ascontiguousarray(data)


def eccentricity_from_axes(a, b):
    """Return eccentricity sqrt(1 - (b/a)^2) from SEP ellipse axes."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    out = np.full(a.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(a) & np.isfinite(b) & (a > 0.0) & (b >= 0.0)
    if np.any(valid):
        ratio = np.clip(b[valid] / a[valid], 0.0, 1.0)
        out[valid] = np.sqrt(1.0 - ratio * ratio)
    return out


def extract_stars(image, nsigma=5.0, min_pixels=5, max_pixels=1000):
    """
    Detect stars in a full image with SEP and return a structured array.

    Detection is performed on a spatially varying background-subtracted image.
    ``nsigma`` is interpreted as a local-noise multiple by supplying SEP with
    the background RMS map. SEP performs its normal filtering, deblending, and
    cleaning. Sources larger than ``max_pixels`` are discarded after extraction.

    The returned array contains one row per accepted source with fields:
    ``x``, ``y``, ``npix``, ``a``, ``b``, ``theta``, ``flag``, and
    ``eccentricity``.
    """
    data = _as_sep_image(image)

    # SEP estimates a 2-D background and 2-D RMS map, which is important for
    # nebulosity/gradients and avoids weighting source moments with sky pedestal.
    background = sep.Background(data)
    data_sub = data - background.back(dtype=np.float32)
    rms = background.rms(dtype=np.float32)

    objects = sep.extract(
        data_sub,
        float(nsigma),
        err=rms,
        minarea=int(min_pixels),
    )

    dtype = np.dtype([
        ("x", "f8"),
        ("y", "f8"),
        ("npix", "i8"),
        ("a", "f8"),
        ("b", "f8"),
        ("theta", "f8"),
        ("flag", "i8"),
        ("eccentricity", "f8"),
    ])

    if len(objects) == 0:
        return np.empty(0, dtype=dtype)

    ecc = eccentricity_from_axes(objects["a"], objects["b"])

    valid = (
        np.isfinite(objects["x"])
        & np.isfinite(objects["y"])
        & np.isfinite(ecc)
        & (objects["npix"] >= int(min_pixels))
        & (objects["npix"] <= int(max_pixels))
    )
    objects = objects[valid]
    ecc = ecc[valid]

    result = np.empty(len(objects), dtype=dtype)
    for name in ("x", "y", "npix", "a", "b", "theta", "flag"):
        result[name] = objects[name]
    result["eccentricity"] = ecc
    return result


def analyze_cell(cell, nsigma=5.0, min_pixels=5, max_pixels=1000):
    """
    Detect stars in one cell and return ``(n_stars, median_eccentricity)``.

    Kept for API compatibility. The CLI detects sources on the *full image*
    instead and assigns source centroids to cells, so stars on grid boundaries
    are not truncated before their shapes are measured.
    """
    stars = extract_stars(
        cell,
        nsigma=nsigma,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    if len(stars) == 0:
        return 0, np.nan
    return len(stars), float(np.median(stars["eccentricity"]))
