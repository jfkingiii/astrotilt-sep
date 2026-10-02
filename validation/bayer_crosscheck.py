"""Bayer-method cross-check (reviewer request).

Compares three ways of measuring star shapes on raw RGGB one-shot-color data:

  cfa    : astrotilt's method — raw mosaic, per-channel sky subtracted (bayer=True)
  green  : green pixels only, interpolated to full resolution (bilinear on the
           G quincunx; R/B sites get the mean of their 4 G neighbours)
  lum    : bilinear-demosaiced R, G, B (per-channel sky subtracted), summed

Stars with any raw pixel >= 65000 within +-3 px of the centroid are dropped for
every method so all three see the same saturation policy.

Usage:
  python validation/bayer_crosscheck.py synthetic
  python validation/bayer_crosscheck.py real <M44 exposure dir> <out.json>
  python validation/bayer_crosscheck.py summary <out.json> [...]
"""

import glob
import json
import os
import sys

import numpy as np
from astropy.io import fits

from astrotilt.stars import extract_stars, orientation_summary

PHASES = [(0, 0), (0, 1), (1, 0), (1, 1)]  # R, G1, G2, B for RGGB
SAT = 65000.0


def conv3(img, k):
    """3x3 correlation with edge replication (numpy only)."""
    p = np.pad(img, 1, mode="edge")
    H, W = img.shape
    out = np.zeros_like(img)
    for dy in range(3):
        for dx in range(3):
            if k[dy, dx]:
                out += k[dy, dx] * p[dy:dy + H, dx:dx + W]
    return out


def equalize(raw):
    out = raw.astype(np.float32).copy()
    for dy, dx in PHASES:
        out[dy::2, dx::2] -= np.median(out[dy::2, dx::2])
    return out


def masks(shape):
    m = {}
    for name, (dy, dx) in zip(("R", "G1", "G2", "B"), PHASES):
        a = np.zeros(shape, np.float32)
        a[dy::2, dx::2] = 1
        m[name] = a
    m["G"] = m["G1"] + m["G2"]
    return m


K_G = np.array([[0, 1, 0], [1, 4, 1], [0, 1, 0]], np.float32) / 4
K_RB = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], np.float32) / 4


def green_image(raw):
    eq = equalize(raw)
    m = masks(eq.shape)
    return conv3(eq * m["G"], K_G)


def lum_image(raw):
    eq = equalize(raw)
    m = masks(eq.shape)
    return conv3(eq * m["R"], K_RB) + conv3(eq * m["G"], K_G) + conv3(eq * m["B"], K_RB)


def measure(raw, method):
    if method == "cfa":
        s = extract_stars(raw, bayer=True)
    elif method == "green":
        s = extract_stars(green_image(raw))
    else:
        s = extract_stars(lum_image(raw))
    if len(s):
        H, W = raw.shape
        keep = np.ones(len(s), bool)
        for i, (x, y) in enumerate(zip(s["x"], s["y"])):
            xi, yi = int(round(x)), int(round(y))
            keep[i] = raw[max(yi - 3, 0):yi + 4, max(xi - 3, 0):xi + 4].max() < SAT
        s = s[keep]
    return s


def grid(s, W, H):
    ci = np.minimum(s["y"] // (H // 3), 2).astype(int) * 3 + np.minimum(s["x"] // (W // 3), 2).astype(int)
    med = [float(np.median(s["eccentricity"][ci == k])) if np.any(ci == k) else np.nan for k in range(9)]
    z = [complex(np.mean(s["eccentricity"][ci == k] * np.exp(2j * s["theta"][ci == k])))
         if np.any(ci == k) else complex(np.nan) for k in range(9)]
    return med, z


METHODS = ("cfa", "green", "lum")


# --------------------------------------------------------------------------- synthetic
def synthetic():
    from common import make_field

    rng = np.random.default_rng(8)
    # CA case: R and B PSFs 1.4x wider than G (uncorrected refractor), round in all.
    resp = (0.8, 1.0, 1.0, 0.6)
    print("Synthetic RGGB, per-channel sky offsets, S/N ~300 (G channel), random theta")
    print(f"{'case':38s} {'e_true':>6} | " + " ".join(f"{m:>12}" for m in METHODS))
    for case, ca in (("no chromatic aberration", 1.0), ("R,B PSF 1.4x wider (CA)", 1.4)):
        for e in (0.0, 0.5):
            for fwhm in (3.0, 5.0):
                seed = int(rng.integers(1 << 30))
                chans = []
                for scale in (ca, 1.0, 1.0, ca):
                    clean, truth = make_field(np.random.default_rng(seed), 14, 50, fwhm * scale, e, 1e6,
                                              sky=0.0, sky_sigma=1e-9)
                    chans.append(clean.astype(np.float64))
                norm = 300 * 12 * np.sqrt(4 * np.pi * (fwhm / 2.3548) ** 2) * len(truth) / chans[1].sum()
                raw = np.empty_like(chans[0])
                for k, ((dy, dx), sky) in enumerate(zip(PHASES, (620, 730, 730, 650))):
                    raw[dy::2, dx::2] = chans[k][dy::2, dx::2] * norm * resp[k] + sky
                raw = (raw + rng.normal(0, 12, raw.shape)).astype(np.float32)
                res = []
                for m in METHODS:
                    s = measure(raw, m)
                    z = np.mean(s["eccentricity"] * np.exp(2j * s["theta"]))
                    res.append(f"{np.median(s['eccentricity']):.3f} ({abs(z):.2f})")
                print(f"{case + f', FWHM {fwhm:.0f}':38s} {e:6.2f} | " + " ".join(f"{r:>12}" for r in res))
    print("values: median e (|common-direction vector|); ~0.02 expected for random orientation")


# --------------------------------------------------------------------------- real data
def real(directory, out_path):
    files = sorted(glob.glob(os.path.join(directory, "*.fits")))
    out = {m: [] for m in METHODS}
    for f in files:
        raw = fits.getdata(f).astype(np.float32)
        H, W = raw.shape
        for m in METHODS:
            s = measure(raw, m)
            med, z = grid(s, W, H)
            o = orientation_summary(s, W, H)
            out[m].append({"file": os.path.basename(f), "n": len(s), "med": med,
                           "z": [[c.real, c.imag] for c in z], **o})
    with open(out_path, "w") as fh:
        json.dump({"dir": directory, "results": out}, fh)


def summary(paths):
    for p in paths:
        d = json.load(open(p))
        print(f"\n=== {os.path.basename(d['dir'].rstrip('/'))}  ({len(d['results']['cfa'])} subs)")
        grids = {}
        for m in METHODS:
            rows = d["results"][m]
            med = np.nanmedian(np.array([r["med"] for r in rows]), axis=0)
            z = np.nanmean(np.array([[complex(*c) for c in r["z"]] for r in rows]), axis=0)
            fc = np.array([r["common_e"] * np.exp(2j * np.radians(r["common_theta_deg"])) for r in rows])
            radial = np.median([r["radial_e"] for r in rows])
            grids[m] = med
            corners, edges = med[[0, 2, 6, 8]], med[[1, 3, 5, 7]]
            print(f"  {m:5s} stars/sub {np.mean([r['n'] for r in rows]):5.0f} | median-e grid "
                  + " / ".join(" ".join(f"{v:.2f}" for v in med[3 * r:3 * r + 3]) for r in range(3))
                  + f" | ctr {med[4]:.2f} edge {np.mean(edges):.2f} corner {np.mean(corners):.2f}"
                  + f" | radial {radial:+.3f} | common per-sub {np.mean(np.abs(fc)):.3f}, "
                  f"avg {abs(fc.mean()):.3f}@{np.degrees(np.angle(fc.mean())) / 2:+.0f}")
            print(f"        corner angles " + " ".join(
                f"{abs(z[k]):.2f}@{np.degrees(np.angle(z[k])) / 2:+4.0f}" for k in (0, 2, 6, 8)))
        for m in ("green", "lum"):
            r = np.corrcoef(grids["cfa"], grids[m])[0, 1]
            print(f"  cfa vs {m:5s}: grid correlation {r:.2f}, mean |diff| {np.mean(np.abs(grids['cfa'] - grids[m])):.3f}")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "synthetic":
        synthetic()
    elif mode == "real":
        real(sys.argv[2], sys.argv[3])
    else:
        summary(sys.argv[2:])
