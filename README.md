# MedAudit

**A framework for auditing the reported performance of diagnostic AI models across modalities and datasets.**

MedAudit measures how much of a reported accuracy may be explained by data leakage, shortcut learning,
duplicated data or covariates, and whether it holds on new participants and new sites.

> **Status.** The article describing MedAudit is currently under peer review. Version 0.3.2 of this repository
> corresponds to the submitted manuscript. Until the article is published, please cite the software archive (see
> [Citation](#citation)); the article reference will be added here upon publication.

MedAudit quantifies selected threats to the performance estimate of a diagnostic classifier from its predictions, its
data splits and basic metadata. It applies to data in which several samples may come from the same participant,
patient or lesion, provided that this grouping unit is identified. Each module reports an effect size with an
interval and a rating; the results are summarized in a one-page report card.

| Module | Question | Main quantity |
|---|---|---|
| A1 Split integrity | Do participants appear on both sides of a split, and how much does it change the estimate? | Gap Δ between conventional and participant-wise performance, with a one-sided non-inflation test (±2 points) |
| A2a Participant and site information | Does the representation identify the participant or the site? | Same-participant neighbor rate against a class-conditional chance level; identity and site probes |
| A2b Signal versus identity | Is the observed score distinguishable from what participant identity alone produces? | Group-level label-permutation test with Monte Carlo interval |
| A3 Data integrity | Are there duplicates within or across datasets, or documented shared participants? | Exact and near-duplicate search with per-dataset calibration |
| A4 Covariates | Do covariates alone predict the label, and does the model add value beyond them? | Covariate-only performance; nested held-out ΔAUC |
| A5 Uncertainty | Are naive error bars too narrow? | Ratio of naive to participant-bootstrap interval widths (descriptive) |
| A6 Transfer | Does performance hold on held-out sites or an external dataset? | Drop from internal to external performance |

Ratings are green, amber, red or inconclusive. The overall verdict is red if any module is red, amber if any is
amber, inconclusive if a threat could neither be detected nor excluded, and green otherwise (A5 is excluded).

## Compliance levels

| Level | Inputs | Executable modules |
|---|---|---|
| 1 | Out-of-fold predictions with sample and participant identifiers, fold and seed | A1 (overlap), A5 |
| 2 | Level 1 + predictions of the same model under participant-wise cross-validation | A1, A5 |
| 3 | Level 2 + the representation used by the model; site and covariate tables when available | A1–A5 |
| 4 | Level 3 + predictions on held-out sites or an external dataset | A1–A6 |

## Installation

```bash
git clone https://github.com/ameur-Bensefia/medaudit.git
cd medaudit
python -m venv .venv && source .venv/bin/activate      # or a conda environment with Python 3.11
pip install -r requirements.txt
python -m pytest -q tests
```

`requirements.txt` pins versions compatible with PyTorch 2.2.2, the last release for Intel macOS (NumPy < 2). On
Apple Silicon or Linux, newer versions also work.

## Audit your own model

```python
import medaudit
from medaudit.schema import Dataset

ds = Dataset(name="my_study", X=features, y=labels, group=participant_ids, sample_id=sample_ids,
             site=site_ids, confounders=covariates_dataframe)          # site and confounders are optional

# prediction tables: columns sample_id, group, y_true, score (+ fold, seed)
results = medaudit.audit(ds, preds_reported, preds_participant_wise, preds_external=None, html="report_card.html")
```

`medaudit.train.run_cv(ds, model, "sample" | "group" | "site")` produces both kinds of predictions for any
scikit-learn-compatible classifier. A complete example on simulated data is in `examples/audit_your_model.py`:

```bash
python examples/audit_your_model.py
```

## Reproducing the benchmark

Datasets are not redistributed. Obtain each one from its original distributor and place it under a common root
folder (see `configs/paths.yaml`):

| Folder | Dataset | Source |
|---|---|---|
| `UCI_PD_Voice` | UCI Parkinson's voice | UCI Machine Learning Repository |
| `UCI-HAR Dataset` | UCI Human Activity Recognition | UCI Machine Learning Repository |
| `pamap2` | PAMAP2 | UCI Machine Learning Repository |
| `mit-bih` | MIT-BIH Arrhythmia Database | PhysioNet |
| `PTB-XL` | PTB-XL | PhysioNet |
| `CHB-MIT` | CHB-MIT Scalp EEG | PhysioNet |
| `HAM10000` | HAM10000 and the ISIC 2018 Task 3 test set | Harvard Dataverse / ISIC Archive |
| `NIH ChestX` | NIH ChestX-ray14 | NIH Clinical Center |
| `chexpert` | CheXpert (small) | Stanford AIMI |
| `ABIDE` | ABIDE I and II with the official phenotypic files | ABIDE / INDI |
| `PaHaW`, `NewHandPD`, `Darwin` | Handwriting datasets | Their respective authors |

Alternative folder names are accepted (for example `mit-bh`, `ptb-xl`, `CHB`); see `medaudit/adapters/_util.py`.

Edit `configs/paths.yaml` (or set `MEDAUDIT_CONFIG` to your own copy), then run the scripts in order:

| Script | Purpose |
|---|---|
| `00_inspect.py` | Check the environment and every dataset folder |
| `01_prepare.py` | Load each dataset and compute its representation (image embeddings are cached) |
| `02_audit.py` | Train the reference models under every protocol and run all modules |
| `03_simulation.py` | Validation by simulation (`--reps`, `--base <dataset>` for the real-feature generator) |
| `04_predictive_validity.py` | Leave-one-site-out analysis of transfer |
| `05_tables_figures.py` | Summary tables and figures |
| `06_benchmark_model.py` | Benchmark mode: evaluate your own classifier on the suite |
| `07_new_participant_validation.py` | Conventional and participant-wise estimates against held-out participants (`--holdouts`) |
| `08_duplicate_calibration.py` | Per-dataset calibration of the duplicate search |
| `09_recompute_modules.py` | Recompute A1, A4 and A5 (and optionally A2 with `--perm`) from saved predictions |
| `10_nested_a4.py` | Nested A4 incremental-value analysis |

`scripts/run_all.sh` runs the main pipeline. All long steps are resumable. Image embedding and the simulation take
several hours on a laptop.

## Citation

The accompanying article is under peer review. Until it is published, please cite the software archive:

> Bensefia, A. (2026). *MedAudit: a framework for auditing the reported performance of diagnostic AI models
> across modalities and datasets* (Version 0.3.2) [Computer software]. Zenodo. https://doi.org/ZENODO-DOI

The same information is available in machine-readable form in `CITATION.cff` (GitHub shows it under "Cite this
repository"). Once the article is published, its reference will be added here, and both should then be cited.

## Author and contact

**Ameur Bensefia**, Computer & Information Sciences Division, Higher Colleges of Technology, Abu Dhabi, United Arab
Emirates. Contact: abensefia@hct.ac.ae. Questions, bug reports and suggestions are welcome through the GitHub issue
tracker.

## License

MIT License. Copyright (c) 2026 Ameur Bensefia. See `LICENSE`.
