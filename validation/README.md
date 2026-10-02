# Validation scripts

Scripts used to validate astrotilt's SEP-based star-shape measurement. Run them
from this directory with the project venv active. They are not part of the
package or the test suite.

| Script | What it checks |
|--------|----------------|
| `accuracy_table.py` | True vs measured eccentricity for synthetic stars at several S/N levels (initial SEP extractor vs original astrotilt) |
| `variants.py` | Candidate measurement fixes vs truth; motivated measuring moments on unfiltered pixels |
| `diag_bias.py`, `diag_gradient.py` | Root-cause checks: filter-kernel rounding bias; background-RMS inflation by steep gradients |
| `hard_cases.py` | Close pairs, saturation, frame edges, nebulosity/gradients, low S/N mix, crowding |
| `field_tests.py` | Grid-boundary stars, sky-level stability, median vs mean across frames, orientation diagnostics |
| `samples_compare.py` | Original vs initial SEP vs fixed on `sample_fits/` with known truth |
| `bayer_synthetic.py` | Effect of raw RGGB mosaics on detection and shape (synthetic) |
| `m44_bayer.py` | Raw vs per-channel-equalized vs 2x2 superpixel on real one-shot-color subs |
| `bayer_crosscheck.py` | CFA-aware vs green-only vs debayered luminance, synthetic (incl. chromatic aberration) and real one-shot-color subs |

`common.py` holds the synthetic star renderer, `sep_initial_extract` (the SEP
extractor as first reviewed, before fixes) and `old_extract` (a per-star port
of the original algorithm). `old_extract` needs a checkout of
https://github.com/jfkingiii/astrotilt at `$ASTROTILT_ORIG` (default
`/tmp/astrotilt-orig`).

The real-data checks were run on raw ASI2600MC subs of M44 (an uncorrected
800 mm refractor at 1/2/5/10 s, and a flattened 910 mm refractor at 15 s).
Those subs are not kept in the repository; pass a directory of your own subs.
