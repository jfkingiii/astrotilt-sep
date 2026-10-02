"""Compare measurement variants vs truth on synthetic stars (motivated the fixes).

A: initial SEP extractor (moments on 3x3-filtered image)
B: sep.extract(filter_kernel=None)
C: same detection, subtract filter-kernel variance (0.5 px^2) from x2, y2
D: same detection, recompute 2nd moments on UNFILTERED pixels in SEP segment

Also reports SEP flux S/N (flux / (rms*sqrt(npix))) so a cut can be chosen.
Usage: python validation/variants.py [gauss|moffat] [fwhm]
"""

import sys

import numpy as np
import sep

from common import make_field, match

profile = sys.argv[1] if len(sys.argv) > 1 else "gauss"
fwhm = float(sys.argv[2]) if len(sys.argv) > 2 else 3.5
ECCS = [0.0, 0.3, 0.5, 0.6, 0.7]
SNRS = [10, 20, 30, 50, 100, 300, 1000]
KVAR = 0.5  # variance (px^2) per axis of SEP's default [1 2 1] x [1 2 1] kernel


def ecc_from_moments(x2, y2, xy):
    t = (x2 + y2) / 2
    d = np.sqrt(((x2 - y2) / 2) ** 2 + xy ** 2)
    l1, l2 = t + d, t - d
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.sqrt(np.clip(1 - l2 / l1, 0, 1))


def measure(img):
    bkg = sep.Background(img)
    d = img - bkg.back()
    rms = bkg.rms()
    o, seg = sep.extract(d, 5.0, err=rms, segmentation_map=True)
    out = {"x": o["x"], "y": o["y"]}
    out["A"] = ecc_from_moments(o["x2"], o["y2"], o["xy"])
    out["C"] = ecc_from_moments(np.maximum(o["x2"] - KVAR, 1e-3), np.maximum(o["y2"] - KVAR, 1e-3), o["xy"])
    # D: unfiltered moments over each segment (positive pixels only)
    eD = np.full(len(o), np.nan)
    for k in range(len(o)):
        y0, y1, x0, x1 = o["ymin"][k], o["ymax"][k] + 1, o["xmin"][k], o["xmax"][k] + 1
        m = seg[y0:y1, x0:x1] == k + 1
        yy, xx = np.nonzero(m)
        w = d[y0:y1, x0:x1][m].astype(float)
        w = np.clip(w, 0, None)
        f = w.sum()
        if f <= 0:
            continue
        cx, cy = (w * xx).sum() / f, (w * yy).sum() / f
        eD[k] = ecc_from_moments((w * (xx - cx) ** 2).sum() / f, (w * (yy - cy) ** 2).sum() / f,
                                 (w * (xx - cx) * (yy - cy)).sum() / f)
    out["D"] = eD
    out["snr"] = o["flux"] / (np.median(rms) * np.sqrt(o["npix"]))
    ob = sep.extract(d, 5.0, err=rms, filter_kernel=None)
    return out, ob


rng = np.random.default_rng(7)
print(f"profile={profile} fwhm={fwhm}  median measured e (A=initial SEP, B=no filter, C=kernel-var corrected, D=unfiltered seg moments)")
print(f"{'e_true':>6} {'SNR':>5} {'isoSNR':>6} | {'A':>5} {'B':>5} {'C':>5} {'D':>5} | B det%")
for e in ECCS:
    for snr in SNRS:
        img, truth = make_field(rng, 14, 40, fwhm, e, snr, profile=profile)
        o, ob = measure(img)
        mi = match(o, truth)
        g = mi >= 0
        mb = match(ob, truth)
        eb = np.sqrt(1 - (ob["b"] / ob["a"]) ** 2)[mb >= 0]
        r = [np.nanmedian(o[k][g]) for k in "A"] + [np.median(eb) if len(eb) else np.nan] + \
            [np.nanmedian(o[k][g]) for k in "CD"]
        print(f"{e:6.2f} {snr:5d} {np.median(o['snr'][g]):6.0f} | " + " ".join(f"{v:5.3f}" for v in r)
              + f" | {100*len(np.unique(mb[mb>=0]))/len(truth):4.0f}", flush=True)
    print()
