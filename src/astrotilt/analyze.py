#!/usr/bin/env python3
"""
analyze_tilt.py — FITS/XISF Tilt Analysis via Star Eccentricity

Measures star eccentricity across a 3×3 grid in each image.
Compares star eccentricity across the field to reveal systematic image-shape differences.
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
from astropy.io import fits
from tqdm import tqdm
from xisf import XISF

from astrotilt.stars import extract_stars, orientation_summary


def load_image(path):
    """
    Load a FITS or XISF file.

    Returns ``(data, bayer)``: a 2D float32 numpy array, and True if the image
    is an un-debayered one-shot-color mosaic (``BAYERPAT`` keyword or XISF
    ``ColorFilterArray``).
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".xisf":
        xisf = XISF(path)
        meta = xisf.get_images_metadata()[0]
        bayer = "ColorFilterArray" in meta or "BAYERPAT" in meta.get("FITSKeywords", {})
        data = xisf.read_image(0)
    else:
        with fits.open(path, memmap=False) as hdul:
            bayer = "BAYERPAT" in hdul[0].header
            data = hdul[0].data
    data = data.astype(np.float32)
    if data.ndim == 3:  # color image — collapse to mono
        if data.shape[2] == 1:
            data = data[:, :, 0]
        else:
            data = data.mean(axis=2)
            bayer = False
    return data, bayer


def load_image_data(path):
    """Load a FITS or XISF file and return a 2D float32 numpy array."""
    return load_image(path)[0]


def analyze_file(image_path, nsigma=5.0, min_pixels=5, max_pixels=1000,
                 min_snr=50.0, saturation=None, verbose=False):
    """
    Analyze a single FITS or XISF file. Returns list of dicts (one per grid cell).

    Sources are detected once on the complete frame with SEP, then assigned to
    3x3 cells by centroid. This avoids cutting stars at cell boundaries before
    their shapes are measured.

    Each row also carries the cell's coherent elongation (magnitude and angle
    of the mean spin-2 vector e*exp(2i*theta)), the frame-level orientation
    summary from ``orientation_summary``, and the frame's median eccentricity
    and median FWHM (``frame_median_e``, ``frame_fwhm``).
    """
    GRID = 3
    filename = os.path.basename(image_path)

    if verbose:
        print(f"  Processing {filename} ...", file=sys.stderr, end="", flush=True)

    data, bayer = load_image(image_path)
    H, W = data.shape
    cell_h = H // GRID
    cell_w = W // GRID

    stars = extract_stars(
        data,
        nsigma=nsigma,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        min_snr=min_snr,
        saturation=saturation,
        bayer=bayer,
    )
    frame = orientation_summary(stars, W, H)
    if len(stars):
        frame_median_e = float(np.median(stars["eccentricity"]))
        # Gaussian-equivalent FWHM from the isophotal second moments; reads
        # somewhat below a fitted FWHM but tracks focus/seeing changes.
        frame_fwhm = float(2.3548 * np.median(np.sqrt(stars["a"] * stars["b"])))
    else:
        frame_median_e = frame_fwhm = np.nan

    rows_out = []
    for cell_row in range(GRID):
        for cell_col in range(GRID):
            r0 = cell_row * cell_h
            r1 = H if cell_row == GRID - 1 else (cell_row + 1) * cell_h
            c0 = cell_col * cell_w
            c1 = W if cell_col == GRID - 1 else (cell_col + 1) * cell_w

            in_cell = (
                (stars["y"] >= r0)
                & (stars["y"] < r1)
                & (stars["x"] >= c0)
                & (stars["x"] < c1)
            )
            cell_stars = stars[in_cell]
            n_stars = len(cell_stars)
            median_ecc = (
                float(np.median(cell_stars["eccentricity"]))
                if n_stars
                else np.nan
            )
            coherent = (
                np.mean(cell_stars["eccentricity"] * np.exp(2j * cell_stars["theta"]))
                if n_stars
                else complex(np.nan, np.nan)
            )

            rows_out.append({
                "filename": filename,
                "cell_row": cell_row,
                "cell_col": cell_col,
                "cell_x_center": (c0 + c1) // 2,
                "cell_y_center": (r0 + r1) // 2,
                "n_stars": n_stars,
                "median_eccentricity": median_ecc,
                "coherent_e": float(abs(coherent)),
                "coherent_theta_deg": float(np.degrees(np.angle(coherent)) / 2.0),
                "frame_common_e": frame["common_e"],
                "frame_common_theta_deg": frame["common_theta_deg"],
                "frame_radial_e": frame["radial_e"],
                "frame_median_e": frame_median_e,
                "frame_fwhm": frame_fwhm,
            })

    if verbose:
        print(f" {len(stars)} stars found", file=sys.stderr)

    return rows_out


def collect_image_files(path_arg):
    """Expand directory or glob pattern to list of FITS/XISF file paths."""
    if os.path.isdir(path_arg):
        files = []
        for ext in ("*.fits", "*.fit", "*.FITS", "*.xisf", "*.XISF"):
            files.extend(glob.glob(os.path.join(path_arg, ext)))
        files = sorted(set(files))
    else:
        files = sorted(glob.glob(path_arg))
    return files


def print_summary_grids(df, verbose=False):
    """
    Print median_eccentricity as 3×3 grids to stderr: the median across subs and
    its difference from the center cell. Mean and std only with verbose.
    """
    grp = df.groupby(["cell_row", "cell_col"])["median_eccentricity"]
    median = grp.median()
    stats = {
        "Median across subs": median,
        "Difference from center (median)": median - median.get((1, 1), np.nan),
    }
    if verbose:
        stats["Mean across subs"] = grp.mean()
        stats["Standard deviation across subs"] = grp.std()

    col_w = 8  # width per cell value
    header = "  ".join(f"col {c}".center(col_w) for c in range(3))
    separator = "  ".join("-" * col_w for _ in range(3))

    for title, series in stats.items():
        signed = title.startswith("Difference")

        def fmt(v):
            return (f"{v:+.3f}" if v != 0 and not np.isnan(v) else f"{v: .3f}") if signed else f"{v:.3f}"

        print(f"\n{title}", file=sys.stderr)
        print(f"         {header}", file=sys.stderr)
        print(f"         {separator}", file=sys.stderr)
        for r in range(3):
            vals = "  ".join(
                fmt(series.get((r, c), float("nan"))).center(col_w)
                for c in range(3)
            )
            print(f"  row {r}  {vals}", file=sys.stderr)

    if verbose:
        print_orientation_summary(df, stats["Median across subs"])


def print_orientation_summary(df, median_grid):
    """
    Print star counts, per-cell coherent elongation and frame-level pattern
    diagnostics to stderr.

    Coherent elongation is the mean spin-2 vector e*exp(2i*theta), averaged over
    subs; angles are degrees CCW from +x toward +y in image pixel axes.
    """
    col_w = 11
    header = "  ".join(f"col {c}".center(col_w) for c in range(3))
    separator = "  ".join("-" * col_w for _ in range(3))

    vectors = df["coherent_e"] * np.exp(2j * np.radians(df["coherent_theta_deg"]))
    coherent = vectors.groupby([df["cell_row"], df["cell_col"]]).mean()
    n_stars = df.groupby(["cell_row", "cell_col"])["n_stars"].median()

    print("\nStars per cell (median across subs)", file=sys.stderr)
    print(f"         {header}", file=sys.stderr)
    print(f"         {separator}", file=sys.stderr)
    for r in range(3):
        vals = "  ".join(f"{n_stars.get((r, c), 0):.0f}".center(col_w) for c in range(3))
        print(f"  row {r}  {vals}", file=sys.stderr)

    print("\nCoherent elongation across subs (e @ angle)", file=sys.stderr)
    print(f"         {header}", file=sys.stderr)
    print(f"         {separator}", file=sys.stderr)
    for r in range(3):
        cells = []
        for c in range(3):
            z = coherent.get((r, c), complex(np.nan, np.nan))
            cells.append(f"{abs(z):.2f} @{np.degrees(np.angle(z)) / 2:+4.0f}".center(col_w))
        print(f"  row {r}  {'  '.join(cells)}", file=sys.stderr)

    frames = df.drop_duplicates("filename")
    common = (frames["frame_common_e"] * np.exp(2j * np.radians(frames["frame_common_theta_deg"]))).mean()
    radial = frames["frame_radial_e"].median()
    grid = np.array([[median_grid.get((r, c), np.nan) for c in range(3)] for r in range(3)])
    left_right = np.nanmean(grid[:, 2]) - np.nanmean(grid[:, 0])
    top_bottom = np.nanmean(grid[2, :]) - np.nanmean(grid[0, :])

    print("\nField pattern", file=sys.stderr)
    print(f"  common elongation      {abs(common):.3f} @ {np.degrees(np.angle(common)) / 2:+.0f} deg"
          "   (same direction everywhere: tracking / guiding / vibration)", file=sys.stderr)
    print(f"  radial elongation     {radial:+.3f}"
          "   (+ radial, - tangential: backfocus / optical aberration)", file=sys.stderr)
    print(f"  median e, col 2 - col 0  {left_right:+.3f}   (gradient along x: tilt / decentering)", file=sys.stderr)
    print(f"  median e, row 2 - row 0  {top_bottom:+.3f}   (gradient along y: tilt / decentering)", file=sys.stderr)


def per_sub_table(df):
    """One row per sub: star count, FWHM, median e, region medians, orientation."""
    out = []
    for filename, rows in df.groupby("filename", sort=False):
        grid = np.full((3, 3), np.nan)
        for r in rows.itertuples():
            grid[r.cell_row, r.cell_col] = r.median_eccentricity
        first = rows.iloc[0]
        out.append({
            "filename": filename,
            "n_stars": int(rows["n_stars"].sum()),
            "fwhm_px": first["frame_fwhm"],
            "median_eccentricity": first["frame_median_e"],
            "center_e": grid[1, 1],
            "edge_e": np.nanmean(grid[[0, 1, 1, 2], [1, 0, 2, 1]]),
            "corner_e": np.nanmean(grid[[0, 0, 2, 2], [0, 2, 0, 2]]),
            "common_e": first["frame_common_e"],
            "common_theta_deg": first["frame_common_theta_deg"],
            "radial_e": first["frame_radial_e"],
            "x_gradient_e": np.nanmean(grid[:, 2]) - np.nanmean(grid[:, 0]),
            "y_gradient_e": np.nanmean(grid[2, :]) - np.nanmean(grid[0, :]),
        })
    return pd.DataFrame(out)


def print_per_sub(table):
    """Print the per-sub table to stdout, one line per sub."""
    name_w = max(len("file"), *(len(f) for f in table["filename"]))
    print(f"{'file':<{name_w}}  stars   FWHM  med_e  center  edges  corners    common     "
          "radial  x_grad  y_grad")
    for r in table.itertuples():
        common = f"{r.common_e:.2f} @{r.common_theta_deg:+4.0f}"
        print(f"{r.filename:<{name_w}}  {r.n_stars:5d}  {r.fwhm_px:5.2f}  {r.median_eccentricity:5.2f}  "
              f"{r.center_e:6.2f}  {r.edge_e:5.2f}  {r.corner_e:7.2f}  {common:>11}  "
              f"{r.radial_e:+6.3f}  {r.x_gradient_e:+6.3f}  {r.y_gradient_e:+6.3f}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze star eccentricity across a 3×3 image grid using SEP."
    )
    parser.add_argument(
        "input",
        help="Directory containing *.fits/*.xisf files, or a glob pattern (e.g. 'data/*.xisf')",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=5.0,
        metavar="NSIGMA",
        help="Detection threshold in sigma above background (default: 5.0)",
    )
    parser.add_argument(
        "--min-pixels",
        type=int,
        default=5,
        metavar="N",
        help="Minimum pixels per star (default: 5)",
    )
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=1000,
        metavar="N",
        help="Maximum pixels per star (default: 1000)",
    )
    parser.add_argument(
        "--min-snr",
        type=float,
        default=50.0,
        metavar="SNR",
        help="Reject stars below this S/N; faint stars read falsely elongated (default: 50)",
    )
    parser.add_argument(
        "--saturation",
        type=float,
        default=None,
        metavar="ADU",
        help="Reject stars with any pixel at or above this raw value (default: off)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        metavar="FILE",
        help="Output CSV file (default: none)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress to stderr",
    )
    parser.add_argument(
        "--per-sub",
        action="store_true",
        help="Print one line per sub to stdout instead of the 3×3 summary grids; "
             "with --output, write that table as the CSV",
    )
    args = parser.parse_args()

    files = collect_image_files(args.input)
    if not files:
        print(f"Error: no FITS/XISF files found at '{args.input}'", file=sys.stderr)
        sys.exit(1)

    n = len(files)
    all_rows = []
    bar = tqdm(files, desc="Analyzing", unit="image", file=sys.stderr, disable=args.verbose)
    for image_path in bar:
        if args.verbose:
            print(f"  Processing {os.path.basename(image_path)} ...", file=sys.stderr, end="", flush=True)
        rows = analyze_file(
            image_path,
            nsigma=args.threshold,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
            min_snr=args.min_snr,
            saturation=args.saturation,
            verbose=False,
        )
        if args.verbose:
            total_stars = sum(r["n_stars"] for r in rows)
            print(f" {total_stars} stars found", file=sys.stderr)
        all_rows.extend(rows)

    print(f"\nAnalyzed {n} file(s)", file=sys.stderr)

    if args.per_sub:
        table = per_sub_table(pd.DataFrame(all_rows))
        print_per_sub(table)
        if args.output:
            table.to_csv(args.output, index=False)
        return

    df = pd.DataFrame(all_rows, columns=[
        "filename", "cell_row", "cell_col",
        "cell_x_center", "cell_y_center",
        "n_stars", "median_eccentricity",
        "coherent_e", "coherent_theta_deg",
        "frame_common_e", "frame_common_theta_deg", "frame_radial_e",
    ])

    print_summary_grids(df, verbose=args.verbose)

    if args.output:
        df.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
