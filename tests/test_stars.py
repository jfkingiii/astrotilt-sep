"""Synthetic-star regression tests for astrotilt.stars."""

import numpy as np
import pytest

from astrotilt.stars import extract_stars, orientation_summary

SKY, SKY_SIGMA = 1000.0, 10.0
OVERSAMPLE = 5


def star_stamp(fwhm, ecc, theta, dx, dy, half=12):
    """Unit-flux pixel-integrated elliptical Gaussian; fwhm along the major axis."""
    n = 2 * half + 1
    s = (np.arange(n * OVERSAMPLE) + 0.5) / OVERSAMPLE - 0.5 - half
    yy, xx = np.meshgrid(s - dy, s - dx, indexing="ij")
    u = xx * np.cos(theta) + yy * np.sin(theta)
    v = -xx * np.sin(theta) + yy * np.cos(theta)
    sigma = fwhm / 2.3548
    q = np.sqrt(1.0 - ecc ** 2)
    img = np.exp(-0.5 * ((u / sigma) ** 2 + (v / (q * sigma)) ** 2))
    img = img.reshape(n, OVERSAMPLE, n, OVERSAMPLE).sum(axis=(1, 3))
    return img / img.sum()


def star_field(positions, fwhm=3.5, ecc=0.0, theta=None, snr=300.0, shape=(480, 480), seed=0):
    """Image with stars at ``positions``; returns (image, thetas)."""
    rng = np.random.default_rng(seed)
    img = np.full(shape, SKY)
    sigma = fwhm / 2.3548
    flux = snr * SKY_SIGMA * np.sqrt(4 * np.pi * sigma * sigma * np.sqrt(1 - ecc ** 2))
    half = 12
    thetas = []
    for x, y in positions:
        th = rng.uniform(-np.pi / 2, np.pi / 2) if theta is None else theta
        xi, yi = int(np.floor(x)), int(np.floor(y))
        stamp = flux * star_stamp(fwhm, ecc, th, x - xi, y - yi, half)
        y0, x0 = yi - half, xi - half
        ys = slice(max(y0, 0), min(y0 + 2 * half + 1, shape[0]))
        xs = slice(max(x0, 0), min(x0 + 2 * half + 1, shape[1]))
        img[ys, xs] += stamp[ys.start - y0:ys.stop - y0, xs.start - x0:xs.stop - x0]
        thetas.append(th)
    img += rng.normal(0, SKY_SIGMA, shape)
    return img.astype(np.float32), np.array(thetas)


def grid_positions(n=10, spacing=40, seed=1):
    rng = np.random.default_rng(seed)
    return [(spacing * (j + 0.5) + rng.uniform(-0.5, 0.5), spacing * (i + 0.5) + rng.uniform(-0.5, 0.5))
            for i in range(n) for j in range(n)]


def test_round_stars_measure_near_zero():
    img, _ = star_field(grid_positions(), ecc=0.0, snr=1000)
    stars = extract_stars(img)
    assert len(stars) == 100
    assert np.median(stars["eccentricity"]) < 0.15


@pytest.mark.parametrize("fwhm", [2.5, 3.5, 6.0])
@pytest.mark.parametrize("ecc", [0.5, 0.6, 0.7])
def test_elliptical_stars_recover_true_eccentricity(fwhm, ecc):
    # SEP's own a/b are measured on the filtered image and read ~0.1 low here.
    img, _ = star_field(grid_positions(), fwhm=fwhm, ecc=ecc, snr=500)
    stars = extract_stars(img)
    assert len(stars) == 100
    assert np.median(stars["eccentricity"]) == pytest.approx(ecc, abs=0.03)


def test_theta_is_ccw_from_x_axis():
    img, _ = star_field(grid_positions(), ecc=0.6, theta=np.radians(30), snr=500)
    stars = extract_stars(img)
    assert np.degrees(np.median(stars["theta"])) == pytest.approx(30, abs=2)


def test_low_snr_stars_rejected():
    img, _ = star_field(grid_positions(), snr=20)
    assert len(extract_stars(img)) == 0
    assert len(extract_stars(img, min_snr=0)) > 90


def test_edge_truncated_stars_rejected():
    positions = [(1.0, 20.0 + 40 * i) for i in range(10)] + [(240.0, 20.0 + 40 * i) for i in range(10)]
    img, _ = star_field(positions, snr=500)
    stars = extract_stars(img)
    assert len(stars) == 10
    assert np.all(stars["x"] > 200)


def test_saturated_stars_rejected():
    img, _ = star_field(grid_positions(), snr=20000)
    img = np.minimum(img, 20000.0)
    assert len(extract_stars(img)) == 100
    assert len(extract_stars(img, saturation=20000.0)) == 0


def test_star_on_grid_boundary_not_truncated():
    # 480 px frame -> cell boundary at x = 160
    positions = [(160.0 + dx, 30.0 + 40 * i) for i, dx in enumerate(np.linspace(-1, 1, 11))]
    img, _ = star_field(positions, ecc=0.0, snr=1000)
    stars = extract_stars(img)
    assert len(stars) == 11
    assert np.median(stars["eccentricity"]) < 0.15


def test_orientation_summary_patterns():
    positions = grid_positions(n=12)
    common, _ = star_field(positions, ecc=0.5, theta=np.radians(30), snr=500, seed=3)
    s = orientation_summary(extract_stars(common), 480, 480)
    assert s["common_e"] > 0.4
    assert s["common_theta_deg"] == pytest.approx(30, abs=3)

    rng = np.random.default_rng(4)
    img = np.full((480, 480), SKY)
    for x, y in positions:
        phi = np.arctan2(y - 240, x - 240)
        xi, yi = int(x), int(y)
        img[yi - 12:yi + 13, xi - 12:xi + 13] += 3e4 * star_stamp(3.5, 0.5, phi + np.pi / 2, x - xi, y - yi)
    img = (img + rng.normal(0, SKY_SIGMA, img.shape)).astype(np.float32)
    s = orientation_summary(extract_stars(img), 480, 480)
    assert s["common_e"] < 0.1
    assert s["radial_e"] < -0.3  # tangential


def test_bayer_mosaic_sky_offsets_equalized():
    img, _ = star_field(grid_positions(), ecc=0.5, snr=300)
    for (dy, dx), offset in zip([(0, 0), (0, 1), (1, 0), (1, 1)], (-100.0, 0.0, 0.0, -80.0)):
        img[dy::2, dx::2] += offset
    assert len(extract_stars(img)) < 50  # checkerboard inflates the RMS map
    stars = extract_stars(img, bayer=True)
    assert len(stars) == 100
    assert np.median(stars["eccentricity"]) == pytest.approx(0.5, abs=0.03)
