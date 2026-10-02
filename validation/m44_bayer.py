"""Real OSC subs: raw mosaic vs per-channel-equalized vs 2x2 superpixel.

Usage: python m44_bayer.py <subs root> [n subs per exposure]
<subs root> holds one subdirectory per exposure (e.g. 1s/ 2s/ 5s/ 10s/) of raw
RGGB FITS files. The M44 subs used during validation are not in the repo.
"""

import glob
import os
import sys

import numpy as np
import sep
from astropy.io import fits

from astrotilt.stars import extract_stars, orientation_summary


def equalize(img):
    out = img.astype(np.float32).copy()
    for i, j in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        out[i::2, j::2] -= np.median(out[i::2, j::2])
    return out


def superpixel(img):
    img = img.astype(np.float32)
    return img[0::2, 0::2] + img[0::2, 1::2] + img[1::2, 0::2] + img[1::2, 1::2]


def cell_grid(s, W, H):
    ci = np.minimum(s["y"] // (H // 3), 2).astype(int) * 3 + np.minimum(s["x"] // (W // 3), 2).astype(int)
    return np.array([np.median(s["eccentricity"][ci == k]) if np.any(ci == k) else np.nan for k in range(9)])


root = sys.argv[1]
nsub = int(sys.argv[2]) if len(sys.argv) > 2 else 5
for t in sorted(os.listdir(root)):
    files = sorted(glob.glob(os.path.join(root, t, "*.fits")))[:nsub]
    if not files:
        continue
    acc = {k: [] for k in ("raw", "equalized", "superpixel")}
    for f in files:
        raw = fits.getdata(f).astype(np.float32)
        for name, im, sat in (("raw", raw, 65000), ("equalized", equalize(raw), 65000 - 512),
                              ("superpixel", superpixel(raw), None)):
            s = extract_stars(im, saturation=sat)
            H, W = im.shape
            bk = sep.Background(np.ascontiguousarray(im))
            o = orientation_summary(s, W, H)
            g = cell_grid(s, W, H)
            scale = 2.0 if name == "superpixel" else 1.0
            acc[name].append((len(s), bk.globalrms, g[4], np.nanmean(g[[0, 2, 6, 8]]), np.nanmean(g[[1, 3, 5, 7]]),
                              o["common_e"], o["common_theta_deg"], o["radial_e"],
                              scale * 2.3548 * np.median(np.sqrt(s["a"] * s["b"]))))
    print(f"\n=== {t} ({len(files)} subs, means over subs)")
    print(f"  {'method':11s} {'stars':>6} {'bg rms':>7} {'centre':>7} {'edges':>6} {'corners':>7} {'common':>11} {'radial':>7} {'FWHM px':>7}")
    for name, rows in acc.items():
        r = np.array(rows)
        z = np.mean(r[:, 5] * np.exp(2j * np.radians(r[:, 6])))
        print(f"  {name:11s} {r[:,0].mean():6.0f} {r[:,1].mean():7.1f} {r[:,2].mean():7.3f} {r[:,4].mean():6.3f} {r[:,3].mean():7.3f} "
              f"{abs(z):5.3f}@{np.degrees(np.angle(z))/2:+4.0f} {r[:,7].mean():+7.3f} {r[:,8].mean():7.2f}")
