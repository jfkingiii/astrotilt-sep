"""
stars.py — SEP-based star detection and morphology.

SEP (the Python implementation of Source Extractor algorithms) handles
spatially varying background estimation, source detection, filtering and
deblending. Star shapes are then measured from intensity-weighted second
moments of the *unfiltered* background-subtracted pixels in each SEP segment.
SEP's own ``a``/``b``/``theta`` are measured on the detection-filtered image,
which convolves every star with the 3x3 filter kernel and biases small stars
toward round.
"""

import numpy as np
import sep

# Minimum major-axis RMS size (px) for a real star (FWHM ~1.2 px). Hot pixels
# and cosmic rays measure ~0.02-0.15 px; real stars are far larger.
MIN_STAR_SIGMA = 0.5

STAR_DTYPE = np.dtype([
    ("x", "f8"),
    ("y", "f8"),
    ("npix", "i8"),
    ("a", "f8"),
    ("b", "f8"),
    ("theta", "f8"),
    ("flag", "i8"),
    ("snr", "f8"),
    ("peak", "f8"),
    ("eccentricity", "f8"),
])


def _as_sep_image(image):
    """Return a native-endian, C-contiguous float32 2-D array for SEP."""
    data = np.asarray(image, dtype=np.float32)
    if data.ndim != 2:
        raise ValueError(f"Expected a 2-D image, got shape {data.shape}")
    return np.ascontiguousarray(data)


def eccentricity_from_axes(a, b):
    """Return eccentricity sqrt(1 - (b/a)^2) from ellipse semi-axes."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    out = np.full(a.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(a) & np.isfinite(b) & (a > 0.0) & (b >= 0.0)
    if np.any(valid):
        ratio = np.clip(b[valid] / a[valid], 0.0, 1.0)
        out[valid] = np.sqrt(1.0 - ratio * ratio)
    return out


def _segment_moments(values, xx, yy):
    """
    Intensity-weighted centroid and ellipse of one segment.

    Returns ``(cx, cy, a, b, theta)`` with ``a``/``b`` the RMS semi-axes and
    ``theta`` the major-axis angle in radians, counter-clockwise from +x
    toward +y, in [-pi/2, pi/2]. Returns ``None`` if the moments are degenerate.
    """
    w = np.clip(values, 0.0, None)
    total = w.sum()
    if total <= 0.0:
        return None
    cx = np.dot(w, xx) / total
    cy = np.dot(w, yy) / total
    dx = xx - cx
    dy = yy - cy
    x2 = np.dot(w, dx * dx) / total
    y2 = np.dot(w, dy * dy) / total
    xy = np.dot(w, dx * dy) / total

    half_trace = 0.5 * (x2 + y2)
    radius = np.hypot(0.5 * (x2 - y2), xy)
    a2 = half_trace + radius
    b2 = max(half_trace - radius, 0.0)
    if a2 <= 0.0:
        return None
    theta = 0.5 * np.arctan2(2.0 * xy, x2 - y2)
    return cx, cy, np.sqrt(a2), np.sqrt(b2), theta


def extract_stars(image, nsigma=5.0, min_pixels=5, max_pixels=1000,
                  min_snr=50.0, saturation=None, bayer=False):
    """
    Detect stars in a full image with SEP and return a structured array.

    Detection is performed on a spatially varying background-subtracted image.
    ``nsigma`` is relative to SEP's local background-RMS map; because SEP uses
    a matched filter for detection, this is a filtered-image S/N, which reaches
    fainter per-pixel isophotes than a raw ``nsigma`` cut.

    Accepted sources must:

    * have ``min_pixels <= npix <= max_pixels``;
    * not touch the frame edge (SEP ``OBJ_TRUNC``);
    * have S/N >= ``min_snr``, where S/N = segment flux / (rms * sqrt(npix)).
      Faint stars carry a strong upward noise bias in eccentricity, so this cut
      matters for an absolute roundness metric;
    * have no raw pixel at or above ``saturation`` (if given);
    * have a major-axis RMS size ``a >= MIN_STAR_SIGMA`` (rejects hot pixels
      and cosmic rays, which SEP's detection filter spreads over enough
      pixels to pass ``min_pixels``).

    SEP's ``OBJ_MERGED`` flag is *not* used as a cut: deblended neighbours are
    measured normally.

    If ``bayer`` is True the image is treated as a raw 2x2 color-filter mosaic:
    each of the four Bayer phases has its own sky median subtracted before
    background estimation. Otherwise the per-channel sky offsets form a
    checkerboard that SEP counts as noise, inflating the RMS map (2x on 10 s
    OSC subs) and rejecting most stars.

    The returned array has fields ``x``, ``y``, ``npix``, ``a``, ``b``,
    ``theta`` (radians, CCW from +x toward +y), ``flag``, ``snr``, ``peak``
    (max raw pixel value, before any Bayer equalization) and ``eccentricity``.
    """
    data = _as_sep_image(image)
    raw = data
    if bayer:
        data = data.copy()
        for dy in (0, 1):
            for dx in (0, 1):
                data[dy::2, dx::2] -= np.median(data[dy::2, dx::2])

    # SEP estimates a 2-D background and 2-D RMS map, which is important for
    # nebulosity/gradients and avoids weighting source moments with sky pedestal.
    background = sep.Background(data)
    data_sub = data - background.back(dtype=np.float32)
    rms = background.rms(dtype=np.float32)

    # Large frames with nebulosity can exceed SEP's default 300k pixel buffer.
    sep.set_extract_pixstack(max(300_000, data.size // 10))

    objects, segmap = sep.extract(
        data_sub,
        float(nsigma),
        err=rms,
        minarea=int(min_pixels),
        segmentation_map=True,
    )

    result = np.zeros(len(objects), dtype=STAR_DTYPE)
    keep = np.zeros(len(objects), dtype=bool)

    for i, obj in enumerate(objects):
        npix = int(obj["npix"])
        if npix < min_pixels or npix > max_pixels or obj["flag"] & sep.OBJ_TRUNC:
            continue

        y0, y1 = int(obj["ymin"]), int(obj["ymax"]) + 1
        x0, x1 = int(obj["xmin"]), int(obj["xmax"]) + 1
        in_segment = segmap[y0:y1, x0:x1] == i + 1
        yy, xx = np.nonzero(in_segment)
        values = data_sub[y0:y1, x0:x1][in_segment].astype(np.float64)

        peak = float(raw[y0:y1, x0:x1][in_segment].max())
        if saturation is not None and peak >= saturation:
            continue

        noise = float(np.median(rms[y0:y1, x0:x1][in_segment]))
        snr = values.sum() / (noise * np.sqrt(len(values))) if noise > 0 else np.inf
        if snr < min_snr:
            continue

        shape = _segment_moments(values, xx.astype(np.float64), yy.astype(np.float64))
        if shape is None:
            continue
        cx, cy, a, b, theta = shape
        if a < MIN_STAR_SIGMA:
            continue

        result[i] = (x0 + cx, y0 + cy, npix, a, b, theta, obj["flag"], snr, peak,
                     eccentricity_from_axes(a, b))
        keep[i] = True

    return result[keep]


def orientation_summary(stars, width, height):
    """
    Summarize star elongation directions for one frame.

    Each star contributes the spin-2 vector ``e * exp(2i*theta)``, so 180°
    ambiguity in theta cancels correctly. Returns a dict with:

    * ``common_e`` / ``common_theta_deg``: magnitude and angle of the mean
      vector over all stars. Large when all stars point the same way
      (tracking, guiding, wind, vibration).
    * ``radial_e``: mean of ``e * cos(2(theta - phi))`` over stars outside the
      central third of the field, after subtracting the common vector, where
      ``phi`` is the star's position angle from the frame centre. Positive =
      radially elongated, negative = tangentially elongated (backfocus /
      field-curvature / astigmatism patterns). The common vector is removed
      first because on a non-square frame a field-wide elongation would
      otherwise leak into this score (e.g. e=0.4 along x reads +0.09 on 3:2).
    """
    nan = float("nan")
    if len(stars) == 0:
        return {"common_e": nan, "common_theta_deg": nan, "radial_e": nan}

    z = stars["eccentricity"] * np.exp(2j * stars["theta"])
    common = z.mean()

    dx = stars["x"] - width / 2.0
    dy = stars["y"] - height / 2.0
    outer = np.hypot(dx / width, dy / height) > 1.0 / 6.0
    phi = np.arctan2(dy, dx)
    radial = ((z - common) * np.exp(-2j * phi)).real[outer]

    return {
        "common_e": float(abs(common)),
        "common_theta_deg": float(np.degrees(np.angle(common)) / 2.0),
        "radial_e": float(radial.mean()) if len(radial) else nan,
    }


def analyze_cell(cell, nsigma=5.0, min_pixels=5, max_pixels=1000,
                 min_snr=50.0, saturation=None, bayer=False):
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
        min_snr=min_snr,
        saturation=saturation,
        bayer=bayer,
    )
    if len(stars) == 0:
        return 0, np.nan
    return len(stars), float(np.median(stars["eccentricity"]))
