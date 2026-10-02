# astrotilt

Measure star eccentricity across a 3×3 grid in FITS or XISF images. The CLI uses SEP (Source Extractor algorithms) for background estimation, source detection, deblending, and second-moment star-shape measurements, then compares the field regions for systematic image-quality differences such as tilt.

## Installation

Install the CLI:

```bash
pipx install git+https://github.com/jfkingiii/astrotilt
```

Or for development:

```bash
git clone https://github.com/jfkingiii/astrotilt
cd astrotilt
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Usage

```bash
# Analyze all .fits files in a directory, print median grid to stderr
astrotilt samples/

# Save per-cell results to CSV
astrotilt "data/*.fits" --output results.csv

# Show per-file star counts and full summary tables (median, mean, std)
astrotilt samples/ --verbose

# One line per sub (spot trailed, bloated or otherwise bad subs)
astrotilt samples/ --per-sub

# Adjust detection parameters
astrotilt img.fits --threshold 4.0 --min-pixels 3 --max-pixels 500

# Stricter star selection and saturation rejection (16-bit camera)
astrotilt samples/ --min-snr 100 --saturation 65000
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--threshold NSIGMA` | 5.0 | Detection threshold (sigma above background) |
| `--min-pixels N` | 5 | Minimum SEP detection area in pixels |
| `--max-pixels N` | 1000 | Reject detected sources larger than this many pixels |
| `--min-snr SNR` | 50 | Reject stars below this S/N (faint stars read falsely elongated) |
| `--saturation ADU` | off | Reject stars with any raw pixel at or above this value |
| `--output FILE` | — | Write CSV output to FILE |
| `--verbose` | off | Show per-file star counts and full summary tables |
| `--per-sub` | off | Print one line per sub to stdout instead of the 3×3 grids; with `--output`, write that table as the CSV |

## Output

A progress bar is shown during analysis. Afterwards, a 3×3 median eccentricity grid and its difference from the centre cell are printed to stderr:

```
Median across subs
          col 0     col 1     col 2
         --------  --------  --------
  row 0   0.3210    0.2845    0.3501
  row 1   0.2901    0.2634    0.2978
  row 2   0.3412    0.2790    0.3689

Difference from centre (median)
          col 0     col 1     col 2
         --------  --------  --------
  row 0  +0.0576   +0.0211   +0.0867
  row 1  +0.0267    0.0000   +0.0344
  row 2  +0.0778   +0.0156   +0.1055
```

With `--verbose`, mean and standard deviation grids are also printed, along with a per-file star count, a stars-per-cell grid, and orientation diagnostics:

```
Coherent elongation across subs (e @ angle)
            col 0        col 1        col 2
         -----------  -----------  -----------
  row 0   0.06 @  +4   0.55 @ -90   0.04 @ -10
  ...

Field pattern
  common elongation      0.015 @ +1 deg   (same direction everywhere: tracking / guiding / vibration)
  radial elongation     +0.233   (+ radial, - tangential: backfocus / optical aberration)
  median e, col 2 - col 0  -0.008   (gradient along x: tilt / decentering)
  median e, row 2 - row 0  -0.051   (gradient along y: tilt / decentering)
```

Each star contributes the vector `e·exp(2iθ)`, so stars elongated along the same axis add up and randomly oriented stars cancel. Angles are degrees counter-clockwise from +x toward +y in image pixel axes.

With `--output`, a CSV is written with one row per file per cell:

| Column | Description |
|--------|-------------|
| `filename` | FITS file basename |
| `cell_row` / `cell_col` | Grid position (0–2) |
| `cell_x_center` / `cell_y_center` | Pixel center of cell |
| `n_stars` | SEP sources whose centroids fall in the cell |
| `median_eccentricity` | Median second-moment eccentricity (0 = round, 1 = line) |
| `coherent_e` / `coherent_theta_deg` | Magnitude and angle of the cell's mean `e·exp(2iθ)` |
| `frame_common_e` / `frame_common_theta_deg` | Same, over all stars in the frame |
| `frame_radial_e` | Mean `e·cos(2(θ − φ))` outside the central region, after removing the frame's common elongation; + radial, − tangential |

### Per-sub output

With `--per-sub`, astrotilt prints one line per sub to stdout instead of the summary grids:

```
file                 stars   FWHM  med_e  centre  edges  corners    common     radial  x_grad  y_grad
synthetic_0000.fits    168   4.88   0.45    0.33   0.56     0.36   0.05 @  +2  +0.213  -0.039  -0.040
```

| Column | Description |
|--------|-------------|
| `stars` | Stars measured in the sub |
| `FWHM` | Median Gaussian-equivalent FWHM in pixels, from second moments (reads somewhat below a fitted FWHM) |
| `med_e` | Median eccentricity of all stars in the sub |
| `centre` / `edges` / `corners` | Median eccentricity of the centre cell, and mean of the edge-centre and corner cells' medians |
| `common` | The sub's common elongation (e @ angle); large when stars are trailed in one direction |
| `radial` | Radial score (+ radial, − tangential) |
| `x_grad` / `y_grad` | Median e of column 2 minus column 0, and row 2 minus row 0 |

Because eccentricity is a ratio of axes, a fixed trail length stretches small stars more than bloated ones; read `med_e` together with `FWHM`. The summary grids take the median across subs, which hides a few bad subs; `--per-sub` shows them.

With `--output`, the CSV has columns `filename`, `n_stars`, `fwhm_px`, `median_eccentricity`, `centre_e`, `edge_e`, `corner_e`, `common_e`, `common_theta_deg`, `radial_e`, `x_gradient_e`, `y_gradient_e`.

## Measurement method

For each image, astrotilt now:

1. estimates a spatially varying background and noise map with `sep.Background`;
2. subtracts that background;
3. detects/deblends sources with `sep.extract` at the requested sigma threshold;
4. rejects sources that touch the frame edge (SEP `OBJ_TRUNC`), fall below `--min-snr`, contain pixels at or above `--saturation`, or are too sharp to be stars (hot pixels and cosmic rays);
5. computes intensity-weighted second moments of the *unfiltered* background-subtracted pixels in each SEP segment (SEP's own `a`/`b` are measured on its detection-filtered image, which makes small stars look rounder);
6. computes eccentricity as `sqrt(1 - (b/a)^2)`; and
7. assigns each source to a 3×3 cell by its centroid.

Sources are detected on the full frame rather than independently inside each grid cell, so stars crossing a grid boundary are not truncated before their shape is measured.

`--threshold` is relative to SEP's local background RMS on the matched-filtered image, so it reaches fainter per-pixel levels than a raw sigma cut.

### Interpreting eccentricity

Pixel noise makes every star look somewhat elongated, so perfectly round stars never measure 0. Typical median values for synthetic Gaussian stars (FWHM 3.5 px) are:

| true e | S/N 50 | S/N 100 | S/N 300 |
|--------|--------|---------|---------|
| 0.0 | 0.35 | 0.25 | 0.16 |
| 0.3 | 0.35 | 0.31 | 0.30 |
| 0.5 | 0.49 | 0.49 | 0.49 |
| 0.7 | 0.67 | 0.68 | 0.68 |

With the default `--min-snr 50`, a median below about 0.4 means stars are close to round, and true elongations of 0.5 or more are measured accurately. Raising `--min-snr` lowers the noise floor but leaves fewer stars per cell; aim for 20 or more per cell (shown with `--verbose`). Unresolved double stars and very crowded fields push values up.

## Sample data

`sample_fits/` and `sample_xisf/` contain small synthetic 512×512 frames generated by `scripts/make_samples.py`. They have an artificial Gaussian star field with ~180 stars per image. Stars in the edge-centre cells are stretched 1.25× (radially), while centre and corner stars are round apart from ±10% per-axis size jitter. To regenerate:

```bash
python scripts/make_samples.py
```

## Requirements

- Python 3.9+
- FITS or XISF images from an astronomy camera, preferably calibrated. Color XISF images are collapsed to mono by averaging channels.
- Enough stars per frame for reliable eccentricity statistics (typically 10+ per cell)
