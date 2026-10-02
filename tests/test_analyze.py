"""Tests for the per-file/per-sub summaries in astrotilt.analyze."""

import glob
import os

import numpy as np
import pandas as pd

from astrotilt.analyze import analyze_file, per_sub_table

SAMPLES = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "..", "sample_fits", "*.fits")))


def test_per_sub_table_one_row_per_sub():
    rows = []
    for path in SAMPLES:
        rows.extend(analyze_file(path))
    table = per_sub_table(pd.DataFrame(rows))

    assert list(table["filename"]) == [os.path.basename(p) for p in SAMPLES]
    cells = pd.DataFrame(rows).groupby("filename", sort=False)["n_stars"].sum()
    assert list(table["n_stars"]) == list(cells)
    # samples: edge-centre cells stretched radially, centre and corners round
    assert np.all(table["edge_e"] > table["centre_e"] + 0.15)
    assert np.all(table["radial_e"] > 0.15)
    assert np.all((table["fwhm_px"] > 3) & (table["fwhm_px"] < 6))
