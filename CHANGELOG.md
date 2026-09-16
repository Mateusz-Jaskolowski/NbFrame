# Changelog

All notable changes to NbFrame are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Check observed heavy-atom completeness and backbone continuity around CDR3,
  FR2, the RSA key residue, and angle anchors before structure classification.
- Report coordinate-quality details, actual chain/model identity, status, and
  failure reasons. Incomplete inputs return `insufficient_quality` with no
  probability; their reports remain available in multi-chain results and CLI
  JSON/CSV exports, including summary exports.

### Fixed
- Preserve AHo base positions when aligning sequences with insertions; reject
  malformed or insufficiently resolved pre-aligned inputs.
- Check peptide connectivity before calculating CDR3 angles, retain structural
  insertion identities, detect local VH/VL pairs in mixed complexes, and exclude
  hydrogens/deuterium from contacts and RSA.
- Isolate malformed sequence records and honour alignment/scoring batch sizes;
  add an optional `ncpu` / `--ncpu` worker limit.
- Support multi-character chain IDs through mmCIF intermediates and exports,
  and discover all supported structure formats in directory mode.
- Retain required cosine features in directory tables and prevent duplicate
  input basenames from overwriting results or exported structures.
- Report all-filtered CLI runs clearly and reject non-positive progress
  intervals; keep direct renumbering outputs alive until caller cleanup.

### Compatibility notes
- Some structures with unresolved atoms now have predictions withheld despite
  passing framework RMSD filtering. Complete-input model scores are unchanged.
  The structure CLI exits with code 1 when all reported predictions are withheld,
  after writing requested quality reports. See the structure-classifier guide
  for the checks and their limits.
- Sequence alignment output is a 149-column projection onto base AHo positions;
  insertion residues remain in the original input sequence.
- Directory feature `Structure_ID` values and dictionary keys are now relative
  file paths including extensions. Saved AHo filenames include an input-path
  digest; multi-character chain IDs use `.cif`.
- Direct callers of `renumber_structure_to_aho(..., temp_dir=None)` own cleanup
  of the returned directory. High-level APIs still clean up automatically.
- Models, feature coefficients, classification thresholds, sequence deduplication,
  and first-model selection are unchanged.

## [0.3.0] - 2026-06-13

Structure classifier feature overhaul, retrained on an expanded expert-labeled
set. Resolves cases where a CDR3 made a clear, localized contact with FR2 but was
still called *extended* (e.g. the 5ivo crystal vs. its MD trajectory).

### Changed
- **New contact feature `contact_nres`** replaces `contact_density` as a
  classifier input. It is the soft-weighted **count of CDR3 residues contacting
  FR2** (each non-stem CDR3 residue's closest heavy-atom approach to any FR2
  contact residue, AHo 44–55, passed through the logistic switch and summed).
  Unlike `contact_density` it is **not normalized by loop length**, so a
  localized contact in a long, otherwise-extended loop registers fully — matching
  the expert definition "any part of CDR3 contacting FR2 ⇒ kinked". Against a
  contamination-free set of 46 expert labels it separates kinked from extended
  markedly better than `contact_density` (univariate ROC-AUC ≈ 0.95 vs ≈ 0.91).
- **`fr2_rsa_key` narrowed to AHo position 44 only** (was 44 + 54). Position 44
  carries almost all of the RSA signal; position 54 was near-uninformative and
  only diluted the feature. `FR2_KEY_RSA_AHOS` is now `[44]`.
- **Classifier retrained** on the original 100 expert labels **plus 46 new
  expert labels** (135 structures with a clear kinked/extended call). The retrain
  on the harder, ambiguity-enriched set — not the feature swap alone — is what
  aligns the model with the expert definition.
- Model feature vector is now
  `[cos_alpha_N, tau_N, cos_alpha_C, tau_C, contact_nres, fr2_rsa_key]`.

### Added
- `compute_cdr3_fr2_contact_nres` and `cdr3_fr2_residue_min_distances` in
  `nbframe.structure_features` (both exported).
- `contact_nres` is now reported in the per-structure feature output alongside
  the legacy `contact_density` (which remains for interpretability but is no
  longer consumed by the classifier).

### Performance
- Held-out pool (138 structures, no overlap with training): ROC-AUC 0.997,
  accuracy 97.8%.
- 5-fold CV on the training set: ROC-AUC 0.92, accuracy 0.86 (lower than v0.2.0's
  CV because the training set is now enriched with the hardest, most ambiguous
  cases).

## [0.2.0] - 2026-06-11

Robustness-focused structure classifier update, validated on MD trajectories and
an X-ray refinement ensemble.

### Changed
- **Soft contacts**: `contact_density` uses a logistic switching function
  (midpoint 4.5 Å, width 0.4 Å) instead of a hard 4.5 Å cutoff, removing the
  discontinuity that caused single-frame label flips when a contact drifted
  across the cutoff in MD/ensembles. Set `USE_SOFT_CONTACTS = False` to recover
  the legacy binary behavior.
- **Cosine dihedral encoding**: the CDR3 dihedrals `alpha_N` and `alpha_C` are
  fed to the classifier as `cos(angle)` instead of raw degrees, eliminating the
  ±180° branch-cut artifact that flipped predictions on ~2° geometric changes.
  The bond angles `tau_N`/`tau_C` stay linear. Raw `alpha_N`/`alpha_C` are still
  reported for interpretability.

### Performance
- 5-fold CV ROC-AUC 0.970 (up from 0.962), recall 0.83 → 0.90; substantially
  more stable predictions across MD frames and refinement ensembles.

## [0.1.1]

### Fixed
- `contact_density` divided by non-stem CDR3 length to match the trained model.

[0.3.0]: https://github.com/
[0.2.0]: https://github.com/
[0.1.1]: https://github.com/
