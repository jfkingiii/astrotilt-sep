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
| `m44_bayer.py` | Raw vs per-channel-equalized vs 2x2 superpixel on the real `M44/` subs |
| `bayer_crosscheck.py` | CFA-aware vs green-only vs debayered luminance, synthetic (incl. chromatic aberration) and `M44/` subs |

`common.py` holds the synthetic star renderer, `sep_initial_extract` (the SEP
extractor as first reviewed, before fixes) and `old_extract` (a per-star port
of the original algorithm). `old_extract` needs a checkout of
https://github.com/jfkingiii/astrotilt at `$ASTROTILT_ORIG` (default
`/tmp/astrotilt-orig`).
