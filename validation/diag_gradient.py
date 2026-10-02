"""Why does a linear background gradient inflate e?"""
import numpy as np, sep
from common import make_field, match, extract_stars
rng = np.random.default_rng(5)
for amp in (0, 500, 2000, 8000):
    g = lambda H, W: np.linspace(0, amp, W)[None, :] * np.ones((H, 1))
    img, truth = make_field(rng, 14, 36, 3.5, 0.0, 100, gradient=g)
    s = extract_stars(img); mi = match(s, truth)
    bk = sep.Background(img)
    resid = bk.back() - (1000 + g(*img.shape))
    xb = np.digitize(s["x"], [100, 400])
    th = np.degrees(s["theta"])
    print(f"grad {amp:5d} ADU/504px: e med by x-band (edge/mid/edge) = "
          + " ".join(f"{np.median(s['eccentricity'][xb == k]):.3f}" for k in range(3))
          + f" | |theta| med={np.median(np.abs(th)):.0f}deg | bkg resid rms={resid.std():.2f} | rms map med={np.median(bk.rms()):.1f}")
