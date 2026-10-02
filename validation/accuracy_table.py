"""True vs measured eccentricity for synthetic stars (initial SEP vs original).

Usage: python validation/accuracy_table.py [gauss|moffat] [fwhm]
"""

import sys

import numpy as np

from common import sep_initial_extract, make_field, match, old_extract

profile = sys.argv[1] if len(sys.argv) > 1 else "gauss"
fwhm = float(sys.argv[2]) if len(sys.argv) > 2 else 3.5
ECCS = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
SNRS = [10, 20, 30, 50, 100, 300, 1000]
N_SIDE, SPACING = 14, 40

rng = np.random.default_rng(1)
print(f"profile={profile} fwhm={fwhm}px  stars/field={N_SIDE**2}  random theta & subpixel centroids")
print(f"{'e_true':>6} {'SNR':>5} | {'SEP med':>7} {'p16-p84':>11} {'det%':>5} {'dTheta':>6} | {'OLD med':>7} {'p16-p84':>11} {'det%':>5}")
for e in ECCS:
    for snr in SNRS:
        img, truth = make_field(rng, N_SIDE, SPACING, fwhm, e, snr, profile=profile)
        new = sep_initial_extract(img)
        mi = match(new, truth)
        good = mi >= 0
        en = new["eccentricity"][good]
        # theta error (mod pi), only meaningful for elongated stars
        dth = np.angle(np.exp(2j * (new["theta"][good] - truth["theta"][mi[good]]))) / 2
        dth_s = f"{np.degrees(np.median(np.abs(dth))):6.1f}" if e >= 0.3 and len(dth) else "     -"
        old = old_extract(img)
        mo = match(old, truth)
        eo = old["eccentricity"][mo >= 0]

        def fmt(x):
            if len(x) == 0:
                return f"{'-':>7} {'-':>11}"
            p = np.percentile(x, [16, 50, 84])
            return f"{p[1]:7.3f} {p[0]:5.2f}-{p[2]:4.2f}"

        print(f"{e:6.2f} {snr:5d} | {fmt(en)} {100*len(np.unique(mi[good]))/len(truth):5.0f} {dth_s} | "
              f"{fmt(eo)} {100*len(np.unique(mo[mo>=0]))/len(truth):5.0f}", flush=True)
    print()
