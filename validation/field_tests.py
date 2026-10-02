"""Grid-boundary, brightness/background stability, median-vs-mean stability,
and theta pattern tests: initial SEP extractor (v0.2.0 as reviewed) vs fixed astrotilt.stars."""

import numpy as np

from common import extract_stars, sep_initial_extract, make_field, match, old_extract, psf_stamp, snr_to_flux

rng = np.random.default_rng(21)
FWHM = 3.5
H = W = 900


def cell_index(x, y, H=H, W=W):
    return np.minimum((y // (H // 3)).astype(int), 2) * 3 + np.minimum((x // (W // 3)).astype(int), 2)


def realistic_frame(rng, e_of_xy, theta_of_xy, n=900, sky_sigma=10.0, fwhm=FWHM, sky=1000.0):
    """Random positions; power-law S/N distribution (faint stars dominate), like a real sub."""
    img = np.full((H, W), sky)
    xs, ys = rng.uniform(8, W - 8, n), rng.uniform(8, H - 8, n)
    snrs = np.minimum(12 * (1 - rng.uniform(0, 1, n)) ** -1.0, 5000)
    half = 15
    for x, y, sn in zip(xs, ys, snrs):
        e, th = e_of_xy(x, y), theta_of_xy(x, y)
        q = np.sqrt(1 - e * e)
        xi, yi = int(x), int(y)
        st = snr_to_flux(sn, fwhm, q, sky_sigma) * psf_stamp("gauss", fwhm, q, th, x - xi, y - yi, half)
        y0, x0 = yi - half, xi - half
        ys_, xs_ = slice(max(y0, 0), min(y0 + 31, H)), slice(max(x0, 0), min(x0 + 31, W))
        img[ys_, xs_] += st[ys_.start - y0: ys_.stop - y0, xs_.start - x0: xs_.stop - x0]
    img += rng.normal(0, sky_sigma, img.shape)
    return img.astype(np.float32)


# ---------------------------------------------------------------- 1. grid boundary
print("#### 1. stars straddling the x = W/3 grid line (round, SNR 200)")
for off in (0.0, 1.0, 2.0, 4.0, 10.0):
    img = np.full((H, W), 1000.0)
    tr = []
    for i in range(25):
        x, y = W // 3 + off + rng.uniform(-.5, .5), 18 + 35 * i + rng.uniform(-.5, .5)
        half = 15
        st = snr_to_flux(200, FWHM, 1, 10) * psf_stamp("gauss", FWHM, 1, 0, x - int(x), y - int(y), half)
        img[int(y) - half:int(y) + half + 1, int(x) - half:int(x) + half + 1] += st
        tr.append((x, y, 0))
    img = (img + rng.normal(0, 10, img.shape)).astype(np.float32)
    tr = np.array(tr, dtype=[("x", "f8"), ("y", "f8"), ("theta", "f8")])
    res = []
    for f in (sep_initial_extract, extract_stars, old_extract):
        s = f(img)
        g = match(s, tr) >= 0
        res.append(f"{g.sum():2d}/25 e={np.median(s['eccentricity'][g]):.3f}")
    print(f"  offset {off:4.1f}px from line:  initial SEP {res[0]} | fixed {res[1]} | ORIGINAL {res[2]}")

# ------------------------------------------------- 2. brightness / background stability
print("\n#### 2. stability vs sky level and gain-like scaling (round stars, fixed SNR 150)")
for sky, sig in ((100, 10), (1000, 10), (10000, 10), (1000, 40)):
    img, tr = make_field(rng, 20, 45, FWHM, 0.0, 150, sky=sky, sky_sigma=sig)
    a = sep_initial_extract(img)
    c = extract_stars(img)
    print(f"  sky={sky:5d} sigma={sig:2d}: initial SEP e={np.median(a['eccentricity']):.3f} n={len(a)} | "
          f"fixed e={np.median(c['eccentricity']):.3f} n={len(c)}")
print("  same field, star flux x0.3 / x1 / x3 (S/N changes):")
for k in (0.3, 1, 3):
    img, tr = make_field(rng, 20, 45, FWHM, 0.4, 150 * k)
    a, c = sep_initial_extract(img), extract_stars(img)
    print(f"    flux x{k:3.1f} (true e=0.40): initial SEP e={np.median(a['eccentricity']):.3f} | "
          f"fixed e={np.median(c['eccentricity']) if len(c) else np.nan:.3f} n={len(c)}")

# ------------------------------------------------- 3. median vs mean frame-to-frame
print("\n#### 3. frame-to-frame stability, 15 realistic frames (power-law star brightness, 900 stars)")
print("     truth: e=0.0 centre, rising linearly to e=0.5 in the corner cells (tilt-like)")


def e_tilt(x, y):
    r = np.hypot((x - W / 2) / (W / 2), (y - H / 2) / (H / 2)) / np.sqrt(2)
    return 0.5 * r


def th_rand(x, y):
    return rng.uniform(-np.pi / 2, np.pi / 2)


per = {k: [] for k in ("cur_med", "cur_mean", "can_med", "can_mean")}
counts = {"cur": [], "can": []}
for f in range(15):
    img = realistic_frame(rng, e_tilt, th_rand)
    for name, fn in (("cur", sep_initial_extract), ("can", extract_stars)):
        s = fn(img)
        ci = cell_index(s["x"], s["y"])
        per[f"{name}_med"].append([np.median(s["eccentricity"][ci == k]) for k in range(9)])
        per[f"{name}_mean"].append([np.mean(s["eccentricity"][ci == k]) for k in range(9)])
        counts[name].append(len(s))
truth_cell = [np.mean([e_tilt(x, y) for x in np.linspace((k % 3) * 300, (k % 3) * 300 + 300, 20)
                       for y in np.linspace((k // 3) * 300, (k // 3) * 300 + 300, 20)]) for k in range(9)]
print(f"  stars/frame: initial SEP {np.mean(counts['cur']):.0f}, fixed (SNR>=50) {np.mean(counts['can']):.0f}")
print(f"  {'':10s} {'centre':>8} {'edge avg':>8} {'corner':>8} {'corner-centre':>13} {'mean f2f std':>12}")
tc = np.array(truth_cell)
print(f"  {'truth':10s} {tc[4]:8.3f} {tc[[1,3,5,7]].mean():8.3f} {tc[[0,2,6,8]].mean():8.3f} {tc[[0,2,6,8]].mean()-tc[4]:13.3f}")
for k, v in per.items():
    v = np.array(v)
    m = v.mean(axis=0)
    print(f"  {k:10s} {m[4]:8.3f} {m[[1,3,5,7]].mean():8.3f} {m[[0,2,6,8]].mean():8.3f} "
          f"{m[[0,2,6,8]].mean()-m[4]:13.3f} {v.std(axis=0, ddof=1).mean():12.4f}")

# ------------------------------------------------- 4. theta patterns
print("\n#### 4. orientation diagnostics on realistic frames (fixed extractor)")
print("  per-cell spin-2 mean  <e*exp(2i theta)>  -> |coherent e|, angle; field decomposition:")
print("    common  = |mean over all stars of e*exp(2i theta)|")
print("    radial  = mean of  e*cos(2(theta - phi))  (+ = radial, - = tangential), phi = position angle from centre")


def orientation_summary(s):
    z = s["eccentricity"] * np.exp(2j * s["theta"])
    phi = np.arctan2(s["y"] - H / 2, s["x"] - W / 2)
    r = np.hypot(s["x"] - W / 2, s["y"] - H / 2)
    outer = r > 200
    common = np.abs(np.mean(z))
    radial = np.mean((z * np.exp(-2j * phi)).real[outer])
    ci = cell_index(s["x"], s["y"])
    cells = np.array([np.mean(z[ci == k]) for k in range(9)])
    emed = np.array([np.median(s["eccentricity"][ci == k]) for k in range(9)])
    lr = emed[[2, 5, 8]].mean() - emed[[0, 3, 6]].mean()
    tb = emed[[6, 7, 8]].mean() - emed[[0, 1, 2]].mean()
    return common, radial, cells, lr, tb


scen = {
    "round": (lambda x, y: 0.0, th_rand),
    "tracking (e=0.5, theta=30deg everywhere)": (lambda x, y: 0.5, lambda x, y: np.radians(30)),
    "tangential (e=0.5*r, astig/field curvature)": (
        e_tilt, lambda x, y: np.arctan2(y - H / 2, x - W / 2) + np.pi / 2),
    "radial (e=0.5*r, coma/backfocus-like)": (e_tilt, lambda x, y: np.arctan2(y - H / 2, x - W / 2)),
    "gradient L->R (tilt-like, random theta)": (lambda x, y: 0.55 * x / W, th_rand),
}
print(f"  {'scenario':45s} {'common':>7} {'radial':>7} {'dE L->R':>8} {'dE T->B':>8}  per-cell coherent e")
for name, (ef, tf) in scen.items():
    img = realistic_frame(rng, ef, tf, n=1200)
    s = extract_stars(img)
    common, radial, cells, lr, tb = orientation_summary(s)
    cs = " ".join(f"{abs(c):.2f}@{np.degrees(np.angle(c))/2:+4.0f}" for c in cells)
    print(f"  {name:45s} {common:7.3f} {radial:+7.3f} {lr:+8.3f} {tb:+8.3f}  {cs}")
