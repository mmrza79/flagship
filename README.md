# Subject-Independent Gait Phase Detection from Lower-Limb Surface EMG

**Status legend:** **COMPLETED** = infrastructure implemented and unit-tested;
**IN PROGRESS** = blocked on verified data integration; **PLANNED** = not yet
implemented. No experimental results are claimed in this repository.

## Research motivation

Lower-limb surface electromyography (sEMG) provides a non-invasive view of muscle
activation during locomotion. Reliable gait-phase detection from sEMG could support
gait analysis and future rehabilitation-assistance systems. A useful model must
generalize beyond participants seen during training, despite inter-participant
differences in physiology, electrode placement, and signal amplitude.

## Research question

Can gait phases be identified from lower-limb surface EMG signals in a way that
generalizes to previously unseen subjects?

## Proposed hypotheses

- **PLANNED — H1:** Interpretable time- and frequency-domain sEMG features contain
  information that permits above-chance gait-phase classification on held-out subjects.
- **PLANNED — H2:** Performance and error patterns differ across participants,
  motivating participant-level reporting alongside pooled metrics.
- **PLANNED — H3:** Model families may differ in their balance of macro-F1, balanced
  accuracy, and interpretability. No model is presumed superior before evaluation.

Hypotheses will be refined after confirming the dataset's phase definition, class
count, task protocol, and participant structure.

## Project objectives

- **COMPLETED:** Establish reproducible research-software structure.
- **COMPLETED:** Implement configurable offline sEMG preprocessing and initial features.
- **COMPLETED:** Provide leakage-aware subject-grouped evaluation infrastructure.
- **IN PROGRESS:** Integrate and document a real dataset without inferring metadata.
- **PLANNED:** Run, compare, and statistically interpret classical baselines.
- **PLANNED:** Study deep models only after a credible classical benchmark exists.

## Planned methodology

1. Verify dataset documentation, licensing, participant IDs, channels, units,
   sampling frequency, and gait annotations.
2. Inspect recordings without modifying raw files.
3. Define participant-preserving cleaning and windowing rules.
4. Extract interpretable per-channel features.
5. Evaluate classical models using participant-disjoint folds.
6. Report aggregate, per-class, per-participant, and uncertainty-aware results.

All preprocessing that learns dataset statistics (for example, feature scaling) must
be fitted on training folds only. Overlapping windows from one participant or source
recording must never cross a train/test boundary.

## Dataset

**IN PROGRESS — no real dataset is currently present in `data/raw/`.** File format,
directory organization, participant count, participant identifiers, sEMG channel
names, signal units, sampling frequency, timestamps, gait events, and phase labels
are therefore **unknown**.

Place original, unmodified files under `data/raw/`; this directory is ignored by Git
except for its placeholder. Do not commit restricted or large biomedical data. Run
the inspection command below, verify its candidate-field hints against authoritative
dataset documentation, then implement the TODO adapter in `src/data/loader.py`.

## Signal-processing pipeline

**COMPLETED as reusable infrastructure; parameter selection remains IN PROGRESS.**

Available operations are DC-offset removal, configurable Butterworth band-pass
filtering, configurable IIR notch filtering, optional full-wave rectification,
optional signal normalization, and sample-based overlapping window segmentation.
Sampling frequency and filter frequencies default to `null` in configuration because
they are unknown. Invalid Nyquist settings are rejected.

Filters use forward-backward zero-phase filtering for offline analysis. This prevents
phase displacement but uses future samples and is not a causal real-time method. A
future real-time study would require causal filters and separate validation.

## Feature extraction

**COMPLETED as initial infrastructure.** Implemented time-domain features are mean
absolute value (MAV), root mean square (RMS), waveform length (WL), population
variance, zero crossings (ZC), and slope sign changes (SSC). Implemented spectral
features are mean frequency (MNF) and median frequency (MDF), estimated from a
periodogram. MNF and MDF fail explicitly when a valid sampling frequency is absent or
the window has negligible spectral power.

Feature definitions and threshold choices must be fixed before experiments. Each
channel is processed explicitly; the code does not silently flatten multichannel data.

## Subject-independent evaluation strategy

**COMPLETED as infrastructure; experiments are PLANNED.** Leave-One-Subject-Out
(LOSO) cross-validation uses `LeaveOneGroupOut`, with participant ID passed through
the `groups` argument. `GroupKFold` is also supported for later experiments.

Randomly splitting windows is prohibited: windows from the same participant in both
training and testing would create subject leakage and likely inflate performance.
Overlapping windows add another contamination risk and must retain the source
participant/recording group. `StandardScaler` is inside each applicable scikit-learn
`Pipeline`, so it is fitted on training folds only.

## Planned machine-learning baselines

- **PLANNED:** Logistic Regression
- **PLANNED:** Support Vector Machine (SVM)
- **PLANNED:** Random Forest
- **PLANNED, only if justified:** Gradient Boosting or XGBoost
- **OUT OF CURRENT SCOPE:** 1D CNN, LSTM/GRU, CNN-RNN, and Transformers

Current code creates the first three pipelines with conservative defaults. It does not
run them without a verified feature table and does not perform hyperparameter search.

## Evaluation

**COMPLETED as infrastructure; results are PLANNED.** Utilities compute accuracy,
balanced accuracy, macro precision, macro recall, macro-F1, confusion matrix, and
per-class recall/sensitivity. Macro-F1 and balanced accuracy are primary considerations
when classes are imbalanced. All reported model metrics must originate from held-out
participants. Participant imbalance and fold variability will also require analysis.

## Repository structure

```text
.
├── configs/baseline.yaml          # Unknown values remain null
├── data/{raw,interim,processed}/  # Data excluded from Git
├── notebooks/                     # No-data-safe EDA templates
├── results/{figures,tables,models}/
├── scripts/                       # Dataset inspection and grouped baseline CLIs
├── src/
│   ├── data/                      # Conservative inspection/loading interface
│   ├── preprocessing/             # Filters and window segmentation
│   ├── features/                  # Initial interpretable sEMG features
│   ├── models/                    # Classical sklearn pipelines and grouped CV
│   ├── evaluation/                # Multiclass metrics
│   └── visualization/             # Labelled plotting helpers
└── tests/                         # Synthetic mathematical checks only
```

## Reproducibility instructions

Python 3.10 or newer is required. Commands work from the repository root.

Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pytest
python scripts/inspect_dataset.py
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pytest
python scripts/inspect_dataset.py
```

After a verified `data/processed/features.csv` exists with participant and target
columns matching `configs/baseline.yaml`, run one model with:

```bash
python scripts/run_baseline.py --model logistic_regression
```

Use `svm` or `random_forest` for the other prepared baselines. The fixed random seed
is 42 where randomness exists. Paths are repository-relative.

## Current project status

Phases 0–3 infrastructure is **COMPLETED**. Real dataset integration, complete EDA,
and all scientific experiments remain **IN PROGRESS** or **PLANNED**. See
[`PROJECT_STATUS.md`](PROJECT_STATUS.md) for checkboxes.

## Future work

1. Integrate verified dataset schema and provenance.
2. Define gait-phase mapping and window-label policy before feature construction.
3. Complete dataset-backed EDA and quality-control reporting.
4. Run classical LOSO baselines and quantify fold/participant uncertainty.
5. Tune models using nested, group-aware validation if justified.
6. Perform statistical analysis and explainability checks.
7. Consider deep learning only after classical results establish a defensible benchmark.

## Limitations

- No dataset is present, so metadata, preprocessing parameters, labels, and expected
  tensor shapes are unknown.
- Preprocessing functions are mathematically tested on small synthetic signals, not
  validated against biomedical recordings. Synthetic signals are not research data.
- Window-label assignment, artifact rejection, missing-data handling, channel
  harmonization, and participant normalization are not yet specified.
- Filter edge effects and minimum viable recording/window lengths require validation.
- Baseline infrastructure has not produced results; no performance claims are made.
- Real-time suitability is outside the current offline zero-phase pipeline.

## Citation and acknowledgements

**TODO:** Add dataset citation, license, original investigators, funding, software
version/DOI, and project-author citation after dataset selection and repository release.
Do not cite this scaffold as an experimentally validated method.

