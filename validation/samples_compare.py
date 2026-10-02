"""Original vs initial SEP (v0.2.0 as reviewed) vs fixed astrotilt.stars on the bundled sample FITS (truth known
from scripts/make_samples.py: center & corner cells ~round apart from +-10% sigma
jitter; edge-center cells stretched 1.25x along one axis)."""
import glob
import numpy as np
from astropy.io import fits
from common import extract_stars, sep_initial_extract, old_extract

rng = np.random.default_rng(0)
ju = lambda n: rng.uniform(0.9, 1.1, n)
def etrue(fx, fy):
    sx, sy = 1.8 * fx * ju(100000), 1.8 * fy * ju(100000)
    r = np.minimum(sx, sy) / np.maximum(sx, sy); return np.median(np.sqrt(1 - r * r))
truth = np.zeros((3, 3))
for r in range(3):
    for c in range(3):
        cd = max(abs(r - 1), abs(c - 1))
        truth[r, c] = etrue(1 + 0.25 * cd * (c != 1), 1 + 0.25 * cd * (r != 1))
files = sorted(glob.glob("../sample_fits/*.fits"))
H = W = 512
def grids(fn):
    meds, n = [], []
    for f in files:
        img = fits.getdata(f).astype(np.float32)
        s = fn(img)
        ci = np.minimum(s["y"] // (H // 3), 2).astype(int) * 3 + np.minimum(s["x"] // (W // 3), 2).astype(int)
        meds.append([np.median(s["eccentricity"][ci == k]) for k in range(9)]); n.append(len(s))
    return np.array(meds), n
print("truth (median per-star e):\n", np.round(truth, 3))
for name, fn in [("ORIGINAL", old_extract), ("INITIAL SEP", sep_initial_extract),
                 ("FIXED", lambda im: extract_stars(im, saturation=65000))]:
    m, n = grids(fn)
    print(f"\n{name}: stars/frame {n}")
    print(" median across subs:\n", np.round(np.median(m, 0).reshape(3, 3), 3))
    print(" mean across subs:\n", np.round(np.mean(m, 0).reshape(3, 3), 3))
    print(" std across subs:\n", np.round(np.std(m, 0, ddof=1).reshape(3, 3), 3))
    print(f" mean |error| vs truth: {np.mean(np.abs(np.median(m,0).reshape(3,3)-truth)):.3f}")
