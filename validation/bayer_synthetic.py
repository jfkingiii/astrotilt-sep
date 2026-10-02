"""Does measuring on a raw (un-debayered) RGGB mosaic bias eccentricity?

Stars are rendered at full resolution, multiplied by per-channel response
(star colour x QE) on an RGGB mosaic, with a per-channel sky pedestal.
Compared: raw mosaic, per-channel background-equalized mosaic, and 2x2
superpixel (R+G1+G2+B) at half resolution.
"""

import numpy as np

from astrotilt.stars import extract_stars
from common import make_field

RESP = {"white": (1.0, 1.0, 1.0, 1.0), "red (K star)": (1.0, 0.7, 0.7, 0.35), "blue": (0.6, 1.0, 1.0, 1.0)}
SKY = (620.0, 730.0, 730.0, 650.0)  # R, G1, G2, B (like the 10 s M44 subs)


def mosaic(img_noiseless, resp, rng, sigma=12.0):
    out = np.empty_like(img_noiseless)
    for k, (i, j) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
        out[i::2, j::2] = img_noiseless[i::2, j::2] * resp[k] + SKY[k]
    return (out + rng.normal(0, sigma, out.shape)).astype(np.float32)


def superpixel(img):
    return img[0::2, 0::2] + img[0::2, 1::2] + img[1::2, 0::2] + img[1::2, 1::2]


def equalize(img):
    out = img.copy()
    for i, j in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        out[i::2, j::2] -= np.median(img[i::2, j::2])
    return out


rng = np.random.default_rng(5)
print(f"{'colour':14s} {'FWHM':>4} {'e_true':>6} | {'mono ref':>8} {'raw RGGB':>8} {'equalized':>9} {'superpix':>8} | raw common-e @angle")
for colour, resp in RESP.items():
    for fwhm in (3.0, 5.0):
        for e in (0.0, 0.5):
            # noiseless field, then add noise ourselves
            clean, truth = make_field(rng, 16, 48, fwhm, e, 1e6, sky=0.0, sky_sigma=1e-9)
            clean = clean.astype(np.float64)
            # scale to SNR ~300 per star for sigma=12
            clean *= 300 * 12 * np.sqrt(4 * np.pi * (fwhm / 2.3548) ** 2) / clean.sum() * len(truth)
            mono = (clean + 700 + rng.normal(0, 12, clean.shape)).astype(np.float32)
            raw = mosaic(clean, resp, rng)
            res = []
            for im in (mono, raw, equalize(raw), superpixel(raw)):
                s = extract_stars(im)
                res.append((np.median(s["eccentricity"]), len(s), s))
            s = res[1][2]
            z = np.mean(s["eccentricity"] * np.exp(2j * s["theta"]))
            print(f"{colour:14s} {fwhm:4.1f} {e:6.2f} | " + " ".join(f"{m:5.3f}({n:3d})"[:8].rjust(8) for m, n, _ in res[:2])
                  + f" {res[2][0]:9.3f} {res[3][0]:8.3f} | {abs(z):.2f} @{np.degrees(np.angle(z))/2:+.0f}")
