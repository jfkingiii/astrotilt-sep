#!/usr/bin/env python3
"""
make_samples.py — Generate small synthetic FITS and XISF sample files.

Produces 3 synthetic 512×512 uint16 frames per format with artificial
Gaussian stars and background noise. Stars near corners are elongated
to simulate realistic tilt-induced eccentricity variation.

Usage:
    python scripts/make_samples.py
"""

import os

import numpy as np
from astropy.io import fits
from xisf import XISF

OUTDIR_FITS = "sample_fits"
OUTDIR_XISF = "sample_xisf"
N_FILES = 3
HEIGHT, WIDTH = 512, 512
BG_MEAN, BG_SIGMA = 1000.0, 30.0
STARS_PER_CELL = 20
GRID = 3


def add_gaussian_star(image, cy, cx, peak, sigma_y, sigma_x):
    """Add a 2D Gaussian star to `image` in-place (float64 accumulator)."""
    radius = int(max(sigma_y, sigma_x) * 4) + 1
    y0, y1 = max(0, cy - radius), min(HEIGHT, cy + radius + 1)
    x0, x1 = max(0, cx - radius), min(WIDTH, cx + radius + 1)
    ys = np.arange(y0, y1)
    xs = np.arange(x0, x1)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    gaussian = peak * np.exp(
        -0.5 * (((yy - cy) / sigma_y) ** 2 + ((xx - cx) / sigma_x) ** 2)
    )
    image[y0:y1, x0:x1] += gaussian


def make_frame(rng, frame_index):
    """Generate one synthetic 512×512 uint16 image."""
    image = rng.normal(BG_MEAN, BG_SIGMA, (HEIGHT, WIDTH))

    cell_h = HEIGHT // GRID
    cell_w = WIDTH // GRID

    for cell_row in range(GRID):
        for cell_col in range(GRID):
            r0 = cell_row * cell_h
            r1 = HEIGHT if cell_row == GRID - 1 else (cell_row + 1) * cell_h
            c0 = cell_col * cell_w
            c1 = WIDTH if cell_col == GRID - 1 else (cell_col + 1) * cell_w

            # Eccentricity increases toward corners: measure distance from centre
            corner_dist = max(abs(cell_row - 1), abs(cell_col - 1))  # 0 or 1
            sigma_base = 1.8  # ~FWHM 4px
            # Elongate in x for left/right corners, y for top/bottom
            sigma_y = sigma_base * (1.0 + 0.25 * corner_dist * (cell_row != 1))
            sigma_x = sigma_base * (1.0 + 0.25 * corner_dist * (cell_col != 1))

            for _ in range(STARS_PER_CELL):
                cy = int(rng.uniform(r0 + 5, r1 - 5))
                cx = int(rng.uniform(c0 + 5, c1 - 5))
                peak = rng.uniform(5000, 15000)
                # Small per-star jitter in sigma
                sy = sigma_y * rng.uniform(0.9, 1.1)
                sx = sigma_x * rng.uniform(0.9, 1.1)
                add_gaussian_star(image, cy, cx, peak, sy, sx)

    image = np.clip(image, 0, 65535).astype(np.uint16)
    return image


def main():
    os.makedirs(OUTDIR_FITS, exist_ok=True)
    os.makedirs(OUTDIR_XISF, exist_ok=True)

    rng = np.random.default_rng(42)

    for i in range(N_FILES):
        data = make_frame(rng, i)

        fits_path = os.path.join(OUTDIR_FITS, f"synthetic_{i:04d}.fits")
        fits.PrimaryHDU(data).writeto(fits_path, overwrite=True)
        print(f"Wrote {fits_path}  ({os.path.getsize(fits_path) // 1024} KB)")

        xisf_path = os.path.join(OUTDIR_XISF, f"synthetic_{i:04d}.xisf")
        XISF.write(xisf_path, data[:, :, np.newaxis])  # XISF requires (H, W, C)
        print(f"Wrote {xisf_path}  ({os.path.getsize(xisf_path) // 1024} KB)")


if __name__ == "__main__":
    main()
