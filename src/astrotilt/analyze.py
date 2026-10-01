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

from astrotilt.stars import extract_stars


def load_image_data(path):
    """Load a FITS or XISF file and return a 2D float32 numpy array."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".xisf":
        data = XISF(path).read_image(0)
    else:
        with fits.open(path, memmap=False) as hdul:
            data = hdul[0].data
    data = data.astype(np.float32)
    if data.ndim == 3:  # color image — collapse to mono
        data = data.mean(axis=2)
    return data


def analyze_file(image_path, nsigma=5.0, min_pixels=5, max_pixels=1000, verbose=False):
    """
    Analyze a single FITS or XISF file. Returns list of dicts (one per grid cell).

    Sources are detected once on the complete frame with SEP, then assigned to
    3x3 cells by centroid. This avoids cutting stars at cell boundaries before
    their shapes are measured.
    """
    GRID = 3
    filename = os.path.basename(image_path)

    if verbose:
        print(f"  Processing {filename} ...", file=sys.stderr, end="", flush=True)

    data = load_image_data(image_path)
    H, W = data.shape
    cell_h = H // GRID
    cell_w = W // GRID

    stars = extract_stars(
        data,
        nsigma=nsigma,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )

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

            rows_out.append({
                "filename": filename,
                "cell_row": cell_row,
                "cell_col": cell_col,
                "cell_x_center": (c0 + c1) // 2,
                "cell_y_center": (r0 + r1) // 2,
                "n_stars": n_stars,
                "median_eccentricity": median_ecc,
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
    """Print median_eccentricity as 3×3 grids to stderr. Mean and std only with verbose."""
    grp = df.groupby(["cell_row", "cell_col"])["median_eccentricity"]
    stats = {"Median across subs": grp.median()}
    if verbose:
        stats["Mean across subs"] = grp.mean()
        stats["Standard deviation across subs"] = grp.std()

    col_w = 8  # width per cell value
    header = "  ".join(f"col {c}".center(col_w) for c in range(3))
    separator = "  ".join("-" * col_w for _ in range(3))

    for title, series in stats.items():
        print(f"\n{title}", file=sys.stderr)
        print(f"         {header}", file=sys.stderr)
        print(f"         {separator}", file=sys.stderr)
        for r in range(3):
            vals = "  ".join(
                f"{series.get((r, c), float('nan')):.4f}".center(col_w)
                for c in range(3)
            )
            print(f"  row {r}  {vals}", file=sys.stderr)


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
        "--output",
        type=str,
        default=None,
        metavar="FILE",
        help="Output CSV file (default: stdout)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress to stderr",
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
            verbose=False,
        )
        if args.verbose:
            total_stars = sum(r["n_stars"] for r in rows)
            print(f" {total_stars} stars found", file=sys.stderr)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows, columns=[
        "filename", "cell_row", "cell_col",
        "cell_x_center", "cell_y_center",
        "n_stars", "median_eccentricity",
    ])

    print(f"\nAnalyzed {n} file(s)", file=sys.stderr)

    print_summary_grids(df, verbose=args.verbose)

    if args.output:
        df.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
