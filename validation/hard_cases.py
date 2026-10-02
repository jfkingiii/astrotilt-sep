"""Difficult real-world cases for the initial SEP extractor (v0.2.0 as reviewed; motivated the fixes).

Each case prints median e of matched stars, false/odd detections and SEP flag
counts so the filtering policy can be assessed. True e = 0 everywhere unless noted.
"""

import numpy as np
import sep

from common import sep_initial_extract, make_field, match, psf_stamp, snr_to_flux

rng = np.random.default_rng(11)
FWHM = 3.5


def flagstr(f):
    names = {1: "MERGED", 2: "TRUNC", 4: "DOVERFLOW", 8: "SINGU"}
    return ", ".join(f"{n}={int(np.sum((f & b) > 0))}" for b, n in names.items() if np.any(f & b)) or "none"


def report(name, img, truth, note=""):
    s = sep_initial_extract(img)
    mi = match(s, truth)
    g = mi >= 0
    print(f"\n== {name} {note}")
    print(f"   detections={len(s)} matched={g.sum()} unmatched={np.sum(~g)}  flags: {flagstr(s['flag'])}")
    if g.any():
        print(f"   matched     e median={np.median(s['eccentricity'][g]):.3f} mean={np.mean(s['eccentricity'][g]):.3f}")
    if (~g).any():
        print(f"   unmatched   e median={np.median(s['eccentricity'][~g]):.3f}  npix median={np.median(s['npix'][~g]):.0f}")
    fl = s["flag"] != 0
    if fl.any():
        print(f"   flagged     e median={np.median(s['eccentricity'][fl]):.3f}  unflagged e median={np.median(s['eccentricity'][~fl]):.3f}")
    return s, mi


def add_star(img, x, y, flux, e=0.0, theta=0.0, fwhm=FWHM):
    half = 15
    xi, yi = int(np.floor(x)), int(np.floor(y))
    st = flux * psf_stamp("gauss", fwhm, np.sqrt(1 - e * e), theta, x - xi, y - yi, half)
    H, W = img.shape
    y0, x0 = yi - half, xi - half
    ys = slice(max(y0, 0), min(y0 + 2 * half + 1, H))
    xs = slice(max(x0, 0), min(x0 + 2 * half + 1, W))
    img[ys, xs] += st[ys.start - y0: ys.stop - y0, xs.start - x0: xs.stop - x0]


# 1. Close pairs: separation 2..8 px, equal & 1:5 flux, SNR 200
print("#### 1. close pairs (two round stars, SNR 200)")
for ratio in (1.0, 0.2):
    for sep_px in (3, 4, 5, 6, 8, 10):
        img = np.full((600, 600), 1000.0)
        truth = []
        for i in range(10):
            for j in range(10):
                x, y = 30 + 60 * j + rng.uniform(-.5, .5), 30 + 60 * i + rng.uniform(-.5, .5)
                ang = rng.uniform(0, np.pi)
                add_star(img, x, y, snr_to_flux(200, FWHM, 1, 10))
                add_star(img, x + sep_px * np.cos(ang), y + sep_px * np.sin(ang), ratio * snr_to_flux(200, FWHM, 1, 10))
                truth.append((x, y, 0))
        img = (img + rng.normal(0, 10, img.shape)).astype(np.float32)
        s = sep_initial_extract(img)
        # stars within 6+sep px of a primary
        tr = np.array(truth, dtype=[("x", "f8"), ("y", "f8"), ("theta", "f8")])
        mi = match(s, tr, tol=sep_px + 3)
        g = mi >= 0
        merged = (s["flag"] & sep.OBJ_MERGED) > 0
        print(f"  flux ratio {ratio:3.1f} sep {sep_px:2d}px: {len(s):3d} detections for 200 stars; "
              f"e med={np.median(s['eccentricity'][g]):.3f}  frac>0.5={np.mean(s['eccentricity'][g] > 0.5):.2f}  "
              f"MERGED flag={merged.sum()}  e(unflagged)={np.median(s['eccentricity'][g & ~merged]) if (g & ~merged).any() else np.nan:.3f}")

# 2. Saturated stars: clip at 65535, peak up to 20x saturation
print("\n#### 2. saturated round stars (clipped at 65535)")
for sat_factor in (0.5, 1.5, 3, 10, 30):
    peak = sat_factor * (65535 - 1000)
    img, truth = make_field(rng, 10, 60, FWHM, 0.0, 10)
    img = img.astype(np.float64)
    for x, y, _ in truth:
        add_star(img, x, y, peak * 2 * np.pi * (FWHM / 2.3548) ** 2)
    img = np.clip(img, 0, 65535).astype(np.float32)
    s = sep_initial_extract(img)
    mi = match(s, truth)
    g = mi >= 0
    print(f"  peak={sat_factor:4.1f}x saturation: matched {g.sum()}/100 e med={np.median(s['eccentricity'][g]):.3f} "
          f"npix med={np.median(s['npix'][g]):.0f} flags: {flagstr(s['flag'])}")

# Saturated + elongated: does flat top bias e?
print("  elongated e=0.6, 10x saturated: ", end="")
img, truth = make_field(rng, 10, 60, FWHM, 0.6, 10)
img = img.astype(np.float64)
for x, y, th in truth:
    add_star(img, x, y, 10 * 65535 * 2 * np.pi * 2.21 * 0.8, e=0.6, theta=th)
img = np.clip(img, 0, 65535).astype(np.float32)
s = sep_initial_extract(img)
g = match(s, truth) >= 0
print(f"e med={np.median(s['eccentricity'][g]):.3f} (unsaturated equivalent ~0.53 initial SEP code)")

# 3. Edge stars: centroids 0..8 px from the left edge
print("\n#### 3. stars near frame edge (round, SNR 300)")
for d in (0.5, 1.5, 3, 5, 8, 12):
    img = np.full((400, 400), 1000.0)
    truth = []
    for i in range(12):
        y = 20 + 30 * i + rng.uniform(-.5, .5)
        add_star(img, d, y, snr_to_flux(300, FWHM, 1, 10))
        truth.append((d, y, 0))
    img = (img + rng.normal(0, 10, img.shape)).astype(np.float32)
    tr = np.array(truth, dtype=[("x", "f8"), ("y", "f8"), ("theta", "f8")])
    s = sep_initial_extract(img)
    mi = match(s, tr, tol=3)
    g = mi >= 0
    print(f"  centroid {d:4.1f}px from edge: matched {g.sum()}/12 e med={np.median(s['eccentricity'][g]) if g.any() else np.nan:.3f} "
          f"theta med={np.degrees(np.median(np.abs(s['theta'][g]))) if g.any() else np.nan:5.1f}deg flags: {flagstr(s['flag'][g])}")

# 4. Nebulosity / gradient backgrounds
print("\n#### 4. spatially varying backgrounds (round stars, SNR 100)")


def grad_lin(H, W):
    return np.linspace(0, 2000, W)[None, :] * np.ones((H, 1))


def nebula(H, W):
    yy, xx = np.mgrid[0:H, 0:W]
    out = np.zeros((H, W))
    for cx, cy, r, amp in [(150, 200, 60, 800), (380, 300, 25, 1500), (300, 100, 12, 600), (420, 420, 6, 300)]:
        out += amp * np.exp(-0.5 * ((xx - cx) ** 2 + (yy - cy) ** 2) / r ** 2)
    # filaments
    out += 400 * np.exp(-0.5 * ((yy - 0.6 * xx - 50) / 4) ** 2)
    return out


for name, gfun in [("flat", None), ("linear gradient 0-2000 ADU", grad_lin), ("nebula blobs+filament", nebula)]:
    img, truth = make_field(rng, 14, 36, FWHM, 0.0, 100, gradient=gfun)
    report(name, img, truth)

# 5. Low S/N (round stars) -> covered in accuracy table; here: S/N mix and median of cell
print("\n#### 5. low S/N mix: 50% SNR 15 stars + 50% SNR 200 stars, all round")
img1, t1 = make_field(rng, 14, 40, FWHM, 0.0, 15)
img2, t2 = make_field(rng, 14, 40, FWHM, 0.0, 200, offset=(20, 20))
img = img1 + img2 - 1000 - 0 * img2
img = (img1 + (img2 - 1000)).astype(np.float32)
tr = np.concatenate([t1, t2])
s = sep_initial_extract(img)
mi = match(s, tr)
g = mi >= 0
faint = mi < len(t1)
print(f"   all: e med={np.median(s['eccentricity'][g]):.3f}  faint only={np.median(s['eccentricity'][g & faint]):.3f}  "
      f"bright only={np.median(s['eccentricity'][g & ~faint]):.3f}")

# 6. Crowded field: Poisson star positions, luminosity function
print("\n#### 6. crowded field (round stars, power-law fluxes, random positions)")
for density in (500, 2000, 6000):
    H = W = 600
    img = np.full((H, W), 1000.0)
    n = density
    xs, ys = rng.uniform(0, W, n), rng.uniform(0, H, n)
    snrs = 15 * (1 - rng.uniform(0, 1, n)) ** (-1 / 1.0)  # dN/dS ~ S^-2
    snrs = np.minimum(snrs, 3000)
    for x, y, sn in zip(xs, ys, snrs):
        add_star(img, x, y, snr_to_flux(sn, FWHM, 1, 10))
    img = (img + rng.normal(0, 10, img.shape)).astype(np.float32)
    s = sep_initial_extract(img)
    tr = np.zeros(n, dtype=[("x", "f8"), ("y", "f8"), ("theta", "f8")])
    tr["x"], tr["y"] = xs, ys
    bright = snrs > 100
    tb = tr[bright]
    mi = match(s, tb, tol=1.0)
    g = mi >= 0
    fl = s["flag"] != 0
    print(f"  {n:5d} stars/0.36Mpx: {len(s)} det, e med all={np.median(s['eccentricity']):.3f}  "
          f"bright(SNR>100) matched e med={np.median(s['eccentricity'][g]):.3f}  "
          f"flagged={fl.mean():.2f}  e med unflagged={np.median(s['eccentricity'][~fl]):.3f}")
