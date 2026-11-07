"""
Studio documentation for the Guide tab: detailed methods, procedures, diagrams.
Includes the full application manual (same depth as subject-level metrics notes).
"""
from __future__ import annotations

from typing import Any

FLOW_STEPS = [
    "Data panel",
    "Align / pipeline",
    "Tokenize",
    "Probe or FM export",
    "Explain (SHAP)",
    "Validate (stress)",
]

ARCH_LAYERS = [
    {
        "layer": "Raw inputs",
        "items": [
            "Surveys (PHQ/GAD/energy)",
            "Activity / steps / intensity",
            "Vitals · sleep · HRV",
            "Labs / blood tests",
            "EEG · MEG · fNIRS · fMRI · Spikes/LFP",
            "Stimulus + behaviour logs",
            "Clinical notes",
            "Weather / context",
        ],
    },
    {
        "layer": "Alignment",
        "items": [
            "column maps + validation reports",
            "canonical panel (session_state)",
            "QC missingness / coverage / labels",
            "daily_merge / subject–date",
            "asof_join (labs, max age)",
            "trial_align (pre/post ms, baseline)",
            "multi_subject_align + report",
            "events_to_daily rollups",
        ],
    },
    {
        "layer": "Tokenization",
        "items": [
            "TOKENIZER_REGISTRY",
            "TokenBatch schema",
            "f__ features · y__ labels",
            "e__ embeddings · mask__ missingness",
            "Window / epoch / clip tokens",
        ],
    },
    {
        "layer": "Downstream",
        "items": [
            "RF / XGBoost + GroupKFold",
            "Optuna · calibration",
            "SHAP · importance",
            "FM linear probe",
            "Sequence JSONL · contrastive pairs",
            "Manifest v2 · stress tests · prediction artifacts",
        ],
    },
]

DECISION_PATHS = [
    {
        "goal": "Predict PHQ/GAD from wearables",
        "path": "Data (canonical + QC) → Run pipeline (custom train) → Results (metrics, subject CIs, highest leverage, study report) (subject-wise) → Feature importance → SHAP",
    },
    {
        "goal": "Use trial behaviour as predictors",
        "path": "Stimuli/trials → align → daily rollups → merge daily → train → artifacts",
    },
    {
        "goal": "Export data for transformer pretrain",
        "path": "Tokens → Foundation Model (masks/sequences) → Export FM bundle",
    },
    {
        "goal": "Evaluate a frozen representation",
        "path": "Dataset cards / tokens → FM Probe (PCA or W) → compare subject metrics to RF/XGB",
    },
    {
        "goal": "Test robustness",
        "path": "Export baseline → Dataset cards stress suite → re-probe → compare artifacts",
    },
    {
        "goal": "Neuro-only biomarker exploration",
        "path": "EEG/Audio (or MEG/fNIRS/…) → tokenize → probe → subject-wise metrics",
    },
]

METRIC_GLOSSARY = [
    {"metric": "accuracy", "task": "classification", "meaning": "Share of correct labels (OOF rows)"},
    {"metric": "f1 / f1_weighted", "task": "classification", "meaning": "Precision–recall balance; weighted for class imbalance"},
    {"metric": "auc", "task": "classification", "meaning": "Ranking quality of scores vs labels"},
    {"metric": "mse", "task": "regression", "meaning": "Mean squared error of continuous target"},
    {"metric": "train_error", "task": "both", "meaning": "In-sample error; large gap vs OOF ⇒ overfitting"},
    {"metric": "mean_accuracy (subject)", "task": "classification", "meaning": "Mean of per-subject accuracies (people weighted equally)"},
    {"metric": "std_accuracy (subject)", "task": "classification", "meaning": "Spread of performance across people"},
    {"metric": "calibration", "task": "classification", "meaning": "Whether predicted probabilities match observed frequencies"},
]

SHAP_GLOSSARY = [
    {"plot": "Bar (mean |SHAP|)", "use": "Global ranking of drivers"},
    {"plot": "Beeswarm", "use": "Direction + spread of effects across people/windows"},
    {"plot": "Waterfall", "use": "One sample: path from base value to prediction"},
    {"plot": "Force", "use": "Same local story as waterfall, horizontal layout"},
    {"plot": "Dependence", "use": "SHAP vs feature value (optional interaction color)"},
    {"plot": "Interaction", "use": "Pairwise combined effects beyond main effects"},
]

TOKENIZER_TABLE = [
    {"id": "activity_window_v1", "modality": "wearable", "grain": "day/window", "typical features": "steps, intensity, sleep aggregates"},
    {"id": "eeg_continuous_v1", "modality": "EEG", "grain": "sliding epoch", "typical features": "band power, spectrogram bins, spike rates"},
    {"id": "meg_continuous_v1", "modality": "MEG", "grain": "sliding epoch", "typical features": "mag/grad band power"},
    {"id": "fnirs_continuous_v1", "modality": "fNIRS", "grain": "epoch", "typical features": "HbO/HbR, coupling"},
    {"id": "fmri_continuous_v1", "modality": "fMRI", "grain": "window/TR block", "typical features": "ROI means, FC"},
    {"id": "spikes_lfp_v1", "modality": "spikes/LFP", "grain": "epoch", "typical features": "unit rates, LFP bands"},
    {"id": "audio_clip_v1", "modality": "audio", "grain": "clip", "typical features": "simple acoustic stats"},
    {"id": "notes_bag_v1", "modality": "notes", "grain": "note/day", "typical features": "length, keyword hits"},
]

CONTRASTIVE_TABLE = [
    {"strategy": "temporal_nearby", "positives": "same subject, 1 ≤ |Δt| ≤ gap", "negatives": "random other subjects"},
    {"strategy": "temporal_adjacent", "positives": "|Δt| = 1 only", "negatives": "random other subjects"},
    {"strategy": "session_window", "positives": "sampled within gap", "negatives": "random other subjects"},
    {"strategy": "hard_negative", "positives": "nearby same subject", "negatives": "other subjects, similar sequence position"},
    {"strategy": "easy_negative", "positives": "nearby same subject", "negatives": "pure random other subjects"},
]

PROCEDURES = [
    {
        "name": "First MH classification model",
        "steps": [
            "Data panel: demo or multi-file upload → Build canonical panel → check validation + QC",
            "Sidebar: pick target (e.g. phq9_elevated), RF or XGB, window size, n_splits",
            "Run pipeline → Custom train",
            "Results: row metrics AND subject-wise mean ± std; note artifact_dir",
            "Sidebar: imbalance mode (balanced/focal) if rare elevated labels",
            "Feature importance: impurity + permutation",
            "Optional: Calibration tab → calibrated RF+XGB ensemble",
            "Optional: Ablation one-click modality groups",
            "SHAP: beeswarm then waterfall on a high-risk row",
            "Calibration if using probabilities operationally",
            "Archive metrics.json with any written claim",
        ],
    },
    {
        "name": "Trial-aligned physiology",
        "steps": [
            "Stimuli/trials: simulate or upload stimulus + behaviour + continuous",
            "Set pre/post ms and baseline correct",
            "Run multi-subject align → inspect coverage bars and PEA",
            "Tokenize epochs; optional events_to_daily → merge into daily panel",
            "Train with trial_* features; check subject-level metrics",
        ],
    },
    {
        "name": "FM pretrain package",
        "steps": [
            "Ensure windowed or daily tokens exist",
            "Foundation Model: choose norm (per-subject z-score recommended)",
            "Run preprocess → inspect mask heatmap and sequence strips",
            "Pick contrastive strategy; download pairs if SSL needed",
            "Export: flat + sequence (+ contrastive)",
            "Use sequences.jsonl outside the app for transformer training",
            "FM Probe later with same norm policy from manifest; save artifacts",
        ],
    },
]


def guide_sections() -> list[dict[str, str]]:
    """Full application manual (detailed, parallel to subject-level metrics notes)."""
    return [
        {
            "id": "overview",
            "title": "1. What this studio is (and is not)",
            "body": """
### What it is

A **research studio** that:

1. Aligns many data types onto a common subject–time structure  
2. Turns them into a **shared token schema** (`f__`, `y__`, `e__`, `m__`, `mask__`)  
3. Supports **classic ML** (Random Forest / XGBoost) for mental-health targets  
4. Supports **foundation-model style evaluation** (frozen encoder + linear probe)  
5. Exports **sequences, masks, and contrastive pairs** for external transformer pretraining  
6. Enforces a **research trail**: GroupKFold by subject, subject-level metrics, saved prediction artifacts  

### What it is not

| Not this | Why |
|----------|-----|
| A clinical diagnostic device | Outputs are model scores, not diagnoses |
| An end-to-end large FM trainer | Sequences are **exported**; heavy training is outside Streamlit |
| Full clinical NLP | Notes use bag-of-features, not a clinical language model |
| Proof of causation | SHAP/importance explain the **model**, not biology |

Treat it as a **reproducible analysis and evaluation bench**, not a bedside product.
""",
        },
        {
            "id": "token_contract",
            "title": "2. Central design rule: one token contract",
            "body": """
Every modality is forced into **TokenBatch → flat rows** with prefixes:

| Prefix | Meaning | Used for |
|--------|---------|----------|
| `f__` | Numeric features | Model inputs |
| `y__` | Labels / targets | Training and metrics |
| `e__` | Optional embeddings | FM probe / viz |
| `m__` | Metadata | Filtering, audit |
| `mask__` | 1 = observed, 0 = missing | FM-aware absence |

**Why this matters**

Without a shared schema, EEG scripts, wearable tables, and survey scores never meet cleanly. With it, the same RF/XGB, SHAP, FM probe, and sequence export all read one panel.

**How to read a token table**

- Many `f__*` columns → feature space  
- One or more `y__*` → task label(s)  
- `subject_id` + time (`date` / `t0`) → grouping and order  
- `mask__*` present → missingness was recorded, not only imputed away  
""",
        },
        {
            "id": "data_contract",
            "title": "3a. Data contract · multi-file loaders · QC · canonical panel",
            "body": """
### What this is

Hardened **ingestion contract** so every analysis tab shares one panel:

1. **Multi-file loaders** — surveys, activity, labs, notes (+ optional vitals/sleep/behavior/weight/meds/weather)  
2. **Explicit column maps** — aliases (e.g. `participant_id` → `subject_id`, `phq_9` → `phq9`) with a **validation report** per modality  
3. **Canonical panel** — single `subject_id` × `date` table  
4. **QC dashboard** — missingness, subject coverage, label balance **before** any train  

### Session state (single source of truth)

| Key | Role |
|-----|------|
| `canonical_panel` | Source of truth (subject–date panel) |
| `daily_df` | **Same object** — legacy tabs still work |
| `canonical_validation` | Per-file map / OK / missing required columns |
| `canonical_meta` | n_rows, n_subjects, modalities_ok / failed |
| `qc_report` | Missingness, coverage, label balance package |

**Rule:** train, FM preprocess, and stimuli→daily merge should read this panel — not divergent copies.

### Column maps (why they matter)

Raw files rarely share one naming scheme. Maps define accepted aliases; validation reports show what was resolved.

**How to read a validation report**

| Field | Meaning |
|-------|---------|
| `ok` | Required logical columns found |
| `mapped` | logical → actual column name |
| `missing_required` | e.g. no date column — file skipped or failed |
| `n_rows` / `n_subjects` | Size after load |

### QC dashboard (how to read)

| View | Healthy | Warning |
|------|---------|---------|
| Missingness by column | Low on core features/labels | >50% missing on the training target |
| Days per subject | Comparable coverage | Some subjects with <3 days |
| Missingness over time | Stable | Sudden spikes (device dropout) |
| Label balance / fill | Fill ≫ 30%; classes not extreme | Fill <30% or one class only |

**QC flags** (shown in Data panel): low label fill, fewer than 3 subjects, short subject coverage.

### Procedure

1. **Data panel** → demo **or** upload modalities  
2. **Build canonical panel** (or Generate demo)  
3. Read **validation reports** — fix aliases if `ok=False`  
4. Open **QC dashboard** — resolve flags before **Run pipeline**  
5. Custom train / FM — artifacts still write under `exports/runs/`  

### Analysis habit

Never train on an un-validated upload. Archive `canonical_meta` + QC summary with the same claim as `metrics.json`.
""",
        },
        {
            "id": "order",
            "title": "3. Suggested order of work",
            "body": """
| Step | Tab | What you do | What “good” looks like |
|------|-----|-------------|-------------------------|
| 1 | **Data panel** | Multi-file loaders → column maps → **canonical panel** + **QC** | Validation OK; QC flags clear; `canonical_panel` set |
| 2 | **Run pipeline** | Align, engineer, window; **Custom train** | Metrics + artifact path appear |
| 3 | **Results** | Row metrics **and** subject-wise metrics | Mean subject accuracy, not only pooled |
| 4 | **Feature importance** (Results) | Impurity + permutation | Top features make domain sense |
| 5 | **SHAP** | Beeswarm + waterfall on a case | Direction of effects is plausible |
| 6 | **EEG / Audio / neuro** | Tokenize continuous signals if needed | QC plots, optional EEG probe |
| 7 | **Stimuli / trials** | Events → align → PEA → daily rollups | Coverage bars healthy; PEA shows onset effect |
| 8 | **Foundation Model** | Masks, sequences, contrastive pairs | Heatmaps + pair counts look sane |
| 9 | **FM Probe** | Frozen encoder vs trees | Same splits; subject metrics compared |
| 10 | **Optuna / Calibration** | Tune XGB; check probability quality | Lower train–OOF gap; better reliability |
| 11 | **Dataset cards** | Manifest v2, stress tests | Stressed variants degrade gracefully |
| 12 | **Export / Guide** | Save bundles; re-read methods | Artifacts archived with claims |

**Minimum path for a first MH model:** Data → Custom train → Results (subject-wise) → SHAP  

**Minimum path for FM data:** Tokens exist → Foundation Model → Export (flat + sequence)
""",
        },
        {
            "id": "align",
            "title": "4. Alignment: clocks and people",
            "body": """
### Problem

Surveys are sparse, wearables dense, labs irregular, trials event-locked. Naïve joins throw away most data or invent false concurrency.

### Mechanisms

| Mechanism | When to use | How to read it |
|-----------|-------------|----------------|
| **daily_merge** | Both sides already daily | Check row counts before/after |
| **asof_join** | Labs/weight: last value ≤ day, age ≤ max days | If available flags are always 0, max age too tight |
| **trial_align** | Continuous signal around onset/offset | Empty trials in report = windows too short or bad times |
| **Baseline correction** | ERP-style relative to pre-onset | PEA near 0 before onset if baseline worked |
| **events_to_daily** | Trial RT/accuracy → MH daily predictors | `trial_*` columns on subject–date |

**Why subject–date (or trial) grain matters**

Downstream GroupKFold groups by `subject_id`. Alignment defines what a “row” is; bad alignment creates leakage (future labels) or empty features.
""",
        },
        {
            "id": "tokenize",
            "title": "5. Tokenizers and the registry",
            "body": """
### What they are

Adapters from raw tables/signals → TokenBatch lists. Registered by id (`eeg_continuous_v1`, `activity_window_v1`, `notes_bag_v1`, …).

### Families

| Family | Examples | Grain |
|--------|----------|--------|
| Wearable | activity windows | day / sliding window |
| Neuro | EEG, MEG, fNIRS, fMRI, spikes/LFP | epoch / TR block |
| Behaviour | trial epochs after align | trial |
| Text | notes bag-of-features | note / day |
| Audio | clip stats | clip |

### How to use Tokenizers tab

1. Pick modality / tokenizer id  
2. Encode & preview → token count, feature dim, missingness  
3. Optional YAML for reproducible runs without clicking  

### How to read previews

- Very high missingness on key `f__*` → fix upstream alignment  
- Token count much less than expected → filters or min_samples too strict  
- PCA clusters by subject may mean identity signal (careful with claims)
""",
        },
        {
            "id": "stimuli",
            "title": "6. Stimuli / trials",
            "body": """
### Schemas

- **Stimulus:** trial_id, onset, offset, condition, stimulus_id, intensity, modality  
- **Behaviour:** RT, accuracy, choice, confidence, movement  

### Pipeline

1. Validate & merge behaviour  
2. **Multi-subject align** (pre/post ms, baseline)  
3. Coverage bars + raster + peri-event average  
4. Tokenize and/or **daily rollups** (`trial_mean_RT`, `trial_accuracy_rate`, …)  
5. Optional merge into the MH daily panel  

### How to read the plots

| Plot | Healthy pattern | Warning |
|------|-----------------|--------|
| Coverage stacked bar | Mostly aligned | Many empty → timing/fs mismatch |
| Raster | Structure locked to onset | Pure noise, no lock |
| PEA ± SEM | Clear deflection after 0 ms | Flat → no evoked signal or bad align |
| Subject PEA overlay | Similar shape, scaled | One subject dominates / inverted |
""",
        },
        {
            "id": "ml",
            "title": "7. Classic ML (RF / XGBoost)",
            "body": """
### Split discipline

**GroupKFold by `subject_id`**

- Train and test never share a person in a fold  
- Matches the claim: performance on **new people**  
- Different from random row K-fold (which leaks identity)

### Metrics (row-level)

| Metric | Task | Meaning |
|--------|------|---------|
| accuracy | classification | Share correct (pooled rows) |
| f1 / weighted F1 | classification | Precision–recall; better under imbalance |
| auc | classification | Ranking quality of scores |
| mse | regression | Squared error of continuous target |
| train_error | both | In-sample error; large gap vs OOF ⇒ overfit |

### How to read train_error vs OOF

- Both moderate and close → reasonable fit  
- Train much better than OOF → memorization; simplify or regularize (Optuna)  
- Both poor → weak features or impossible task setup  

### Imbalance (rare high PHQ)

| Mode | Behavior |
|------|----------|
| **balanced** | RF `class_weight=balanced`; XGB `scale_pos_weight`; sample weights on fit |
| **focal** | Focal-style sample weights (minority up-weighted) |
| **none** | Uniform weights |

Use **imbalance summary** in metrics (`scale_pos_weight`, class counts).

### Calibrated ensemble

**Calibration** tab → RF + XGB GroupKFold OOF probs → **isotonic** each → **average**.  
Compare Brier raw vs calibrated; subject-wise artifacts under `ensemble_rf_xgb_*`.

### Ablation automation

**Ablation** tab → **one-click** on current windowed X (modality groups) or CSV path.  
Variants: activity, vitals, labs, sleep, behavior, notes, combinations, all.  
Read the results table + bar chart (AUC/accuracy/MSE by variant).

### Target factory

**Data panel** → score PHQ-9 (9 items 0–3 → 0–27), GAD-7 (7 items → 0–21), energy (0–10) with **validation reports** (out-of-range items).  
Also runs inside canonical panel build when item columns exist.
""",
        },
        {
            "id": "subject_metrics",
            "title": "8. Subject-level metrics (detailed)",
            "body": """
### What they are

Each **row** is often one day/window/token for a person, but the unit of interest is the **person**. Subject-level metrics aggregate quality **per person**, then summarize across people.

### Why they matter

| Row-level (pooled) | Subject-level |
|--------------------|---------------|
| One accuracy over all windows | Accuracy (or MSE) **for each subject**, then mean ± std |
| Subjects with many rows dominate | Each subject counts once |
| Can look good if a few prolific subjects are easy | Shows whether the model works for **new people** |

### What the app computes

For each `subject_id` in OOF predictions:

- **Classification:** accuracy, f1, auc (when scores and both classes exist)  
- **Regression:** mse, rmse  
- **Summary:** n_subjects, mean_*, std_*, median_*, min_*, max_*  

### How to read them

| Pattern | Interpretation |
|--------|----------------|
| High mean, low std | Stable across people |
| High mean, high std | Works for some, fails for others |
| Pooled accuracy ≫ mean subject accuracy | A few large/easy subjects inflate the global number |
| One subject near chance | Data quality, label noise, or true heterogeneity |

### Artifacts (mandatory)

Under `exports/runs/<run_name>_<timestamp>/`:

| File | Content |
|------|---------|
| `predictions.csv` | subject_id, y_true, y_pred, y_score*, fold |
| `per_subject_metrics.csv` | metrics **per subject** |
| `metrics.json` | row metrics + **subject_summary** + config |

Written on Custom train, Optuna final train, EEG probe, FM Probe, and FM compare tree runs.\n\nAlso: bootstrap 95% CIs on subject-mean metrics; leakage_audit.json; eval_protocol in metrics.json; predictions.parquet for paper tables. Sidebar **Eval protocol**: groupkfold / repeated / nested / time_aware.
""",
        },
        {
            "id": "importance",
            "title": "9. Feature importance · highest leverage",
            "body": """
| Method | What it measures | Strength | Weakness |
|--------|------------------|----------|----------|
| Impurity / gain | Split usefulness in trees | Fast, native | Biased to high-cardinality / correlated features |
| Permutation | Drop in score when shuffled | Model-agnostic | Slower; correlated features share blame |
| Absolute correlation | Linear association with y | Simple baseline | Misses nonlinear effects |
| SHAP | Additive attribution | Local + global | Explains model, not causes |

**How to read**

- Same feature top in impurity **and** permutation → stronger evidence  
- High correlation, low model importance → redundant with other features  
- Unstable ranks across subjects/folds → heterogeneous mechanisms  
""",
        },
        {
            "id": "shap_theory",
            "title": "10. SHAP views",
            "body": """
SHAP attributes a prediction so that: **f(x) = base_value + sum of SHAP_i**.

| Plot | Question | How to read |
|------|----------|-------------|
| **Bar (mean |SHAP|)** | What matters on average? | Longer bar → larger average impact |
| **Beeswarm** | Direction + heterogeneity? | Color = feature value; x = push up/down |
| **Waterfall** | Why *this* row? | Base → steps → f(x) |
| **Force** | Same local story | Red vs blue tug of war |
| **Dependence** | Nonlinear effect of one feature? | SHAP vs value; color = interaction |
| **Interactions** | Pairwise synergy? | Combined effect beyond main effects |

**Caveats:** model output units; correlated features share credit; not causal.
""",
        },
        {
            "id": "optuna_cal",
            "title": "11. Optuna and calibration",
            "body": """
### Optuna

Searches XGB hyperparameters (history, importance, parallel coordinates, contours).

**How to read**

- Best trial should reflect **CV/OOF**, not train-only  
- Parameter importance shows which knobs moved the objective  
- After search, **re-train with best params** so artifacts match the chosen model  

### Calibration

Whether predicted probabilities match observed frequencies.

**How to read a calibration curve**

- On diagonal → well calibrated  
- Above/below → over/under confident  
- Isotonic comparison: does post-hoc calibration improve reliability without destroying discrimination (AUC)?  

Useful when scores are used as **risk probabilities**, not only rankings.
""",
        },
        {
            "id": "fm_probe",
            "title": "12. FM linear probe (frozen encoder)",
            "body": """
### Protocol

1. Encode features with a **frozen** map: identity, PCA, random projection, or checkpoint `W`  
2. Train only LogisticRegression / Ridge  
3. Same **GroupKFold by subject** as trees  
4. Save subject-level artifacts like any other run  

### How to read tree vs linear probe

| Pattern | Interpretation |
|---------|----------------|
| Linear ≈ RF/XGB | Task is largely linearly readable in that representation |
| Linear ≪ trees | Need nonlinear head or better encoder |
| PCA helps a lot | Signal lives in a lower-dimensional subspace |
| Checkpoint helps | External FM features carry task signal |

Standard **probe** evaluation for foundation models without full fine-tuning in-app.
""",
        },
        {
            "id": "fm_pretrain",
            "title": "13. Foundation Model tab (pretrain data)",
            "body": """
### Unified timeline

`time_index` within subject + `unified_key` → ordered history per person.

### Missingness masks

Imputation alone hides absence. **mask__ columns** tell a transformer what was not measured.

**How to read mask heatmap:** 1 = observed, 0 = missing; systematic zero bands mean sensor off or sparse labs.

### Normalization

`per_subject_zscore` or `robust` — **policy stored in manifest** so probes can match pretrain scale.

### Contrastive strategies

See glossary table (temporal_nearby, adjacent, session_window, hard/easy negative).

### Exports

- **Flat** → RF/XGB / linear probe  
- **Sequence JSONL** → external transformer  
- **Contrastive pairs** → SSL objectives  
""",
        },
        {
            "id": "stress",
            "title": "14. Dataset cards, manifests, stress tests",
            "body": """
**Manifest v2** records seed, splits, modalities, tokenizer params, norm policy, parent dataset, stress tag.

**Stress tests** (missingness, noise, label noise, subject shift, …) create **child datasets**.

**How to read robustness**

- Re-run the same probe on stressed children  
- Rank change in accuracy / AUC  
- Prefer models that degrade smoothly if real deployment has sparse wearables  
""",
        },
        {
            "id": "ablation_play",
            "title": "15. Ablation and Playground",
            "body": """
### Ablation

Drop modality groups, re-fit, compare metrics.

- Large drop when removing a group → that group carried signal  
- No drop → redundant or no signal  
- Prefer **subject-level** summaries when comparing  

### Playground

Edit top features and watch the score change.

- Direction should match SHAP/dependence  
- Huge jumps from tiny edits → unstable or overfit model  
- Still **model counterfactuals**, not real-world interventions  
""",
        },
        {
            "id": "analysis",
            "title": "16. Analysis playbook and claims",
            "body": """
For any result you trust enough to keep:

1. **n subjects**, **n rows**, window definition, target definition  
2. **Split:** GroupKFold, k, seed  
3. **Row-level** metrics **and** **subject_summary** (mean ± std)  
4. **artifact_dir** path (predictions + per_subject_metrics + metrics.json)  
5. Optional: SHAP top drivers + one waterfall example  
6. Optional: calibration if probabilities are used  
7. Limitations: correlation ≠ causation; wearable missingness; label noise  

**Comparing FM probe vs RF:** same subjects philosophy; compare `subject_summary.mean_auc` (or accuracy), not only pooled rows.

**Using artifacts**

1. Results → subject KPI + bar chart  
2. Flag subjects near chance  
3. Archive `metrics.json` with written claims  
4. Compare two runs by joining `per_subject_metrics` on `subject_id`  
""",
        },
        {
            "id": "tabs",
            "title": "17. Tab map (quick reference)",
            "body": """
| Tab | Purpose |
|-----|---------|
| Overview | Stages, outcomes, status pills |
| Data panel | Multi-file loaders, validation, QC, **canonical panel** |
| Run pipeline | Align, window, Custom train |
| Results | Metrics, subject-wise, importance, artifacts |
| Ablation | Modality drop studies |
| EEG / Audio | Neuro continuous tokens + viz |
| Optuna | XGB search + plots |
| Calibration | Probability reliability |
| SHAP | Global/local explanations |
| Interactions | Pairwise SHAP |
| Playground | What-if feature edits |
| Dataset cards | Manifest v2 + stress |
| Tokenizers | Registry preview + YAML |
| Stimuli / trials | Events, PEA, daily rollups |
| FM Probe | Frozen encoder + linear head |
| Foundation Model | Masks, sequences, SSL pairs |
| Export | Downloads + FM bundle |
| Guide | Methods, diagrams, this manual |
| Settings | Session / paths |
""",
        },
        {
            "id": "mental_model",
            "title": "18. Mental model",
            "body": """
```text
Raw multimodal files (surveys · activity · labs · notes · …)
        ↓
   Column maps + validation reports
        ↓
   **Canonical panel** (subject_id × date) + QC dashboard
        ↓
   Align extras (asof / trial) as needed
        ↓
   Tokenizers → f__ / y__ / mask__
        ↓
   ┌────────────────┬────────────────────┐
   ↓                ↓                    ↓
Flat panel    Sequence JSONL      Contrastive pairs
   ↓                ↓                    ↓
RF/XGB · FM probe   External FM train         SSL
   ↓
Subject-wise metrics + artifacts + SHAP
```
""",
        },
        {
            "id": "failures",
            "title": "19. Common failure modes",
            "body": """
| Symptom | Likely cause | What to do |
|---------|--------------|------------|
| Upload builds empty panel | Missing subject_id/date aliases | Check validation `missing_required`; rename columns |
| Great pooled accuracy, poor mean subject accuracy | Few easy subjects dominate rows | Trust subject-level; check per_subject table |
| XGB class errors | Label encoding / rare classes | Class filters, fewer classes, more data |
| Empty trials | Bad onset/fs or min_samples | Fix times; lower min_samples |
| SHAP contradicts domain knowledge | Leakage, wrong target, collinearity | Leakage check; dependence plots |
| FM masks all zero/one | Preprocess not run or no missingness | Run Foundation Model preprocess |
| Train much better than OOF | Overfit | Simpler model, Optuna regularize, more subjects |
""",
        },
        {
            "id": "bottom_line",
            "title": "20. Bottom line",
            "body": """
This app is built so that **mental-health prediction**, **neuro tokenization**, and **foundation-model data prep** share one disciplined path:

- **Subject** is the generalization unit  
- **Tokens** are the common language  
- **Artifacts** are the evidence  
- **SHAP/importance** explain the model, not the world  
- **FM exports** hand off to real sequence models; **FM Probe** scores representations fairly against trees  

**Five habits**

1. Always build/refresh the **canonical panel** and clear **QC flags** before train  
2. Always **GroupKFold by subject** · leave **hard leakage/QC gates** on in Settings  
3. Always read **subject-level** metrics  
4. Always keep the **run artifact folder** · use **paper_claim** and subject CIs in the study report  
5. Always separate **model explanation** from **causal claim**  
""",
        },
        {
            "id": "fm_train_path",
            "title": "Foundation-model path (exports → real training)",
            "body": """
### Pipeline

```text
Foundation Model tab → sequences.jsonl (+ masks)
        ↓
train_fm_encoder (Transformer or TCN)
  · masked feature modeling
  · optional contrastive loss
        ↓
W.npy · encoder.pt · subject_embeddings.csv
        ↓
FM Probe: encoder=checkpoint → linear head
        ↓
Benchmark: linear probe vs RF (same GroupKFold subjects)
```

### Code

| Piece | Location |
|-------|----------|
| Dataset | `mf/fm/dataset.py` · `SequenceJsonlDataset` |
| Models | `mf/fm/models.py` · Tiny Transformer / TCN |
| Train | `mf/fm/train.py` · `train_fm_encoder` |
| CLI | `scripts/train_fm_min.py` |
| Probe | FM Probe tab · `.npy` / `.npz` / `.pt` |

### CLI

```bash
PYTHONPATH=src python scripts/train_fm_min.py \
  --sequences exports/.../sequences.jsonl \
  --out exports/fm_ckpt --model transformer --epochs 5
```

Requires `pip install torch`. This is a **minimal** pretrain for the studio path — not a full foundation model.

### Highest-leverage evaluation discipline

- **Hard leakage gate** (Settings): train refused if GroupKFold subject overlap
- **Hard QC gate**: min subjects / label fill
- **Label prevalence** alerts for rare elevated PHQ
- **Input snapshot** (canonical panel + QC) saved with every custom train
- **paper_claim** string in metrics + HTML/PDF report (subject-mean AUC ± CI)
- FM: auto-discover sequences.jsonl · subject holdout in pretrain · feature align to W
- **Results → Highest leverage features**: combined rank (impurity + corr + SHAP + optional permutation)

""",
        },
        {
            "id": "neuro_product",
            "title": "Neuro stack depth · explainability & product",
            "body": """
### Neuro stack depth

| Capability | Status | Where |
|------------|--------|--------|
| **Full BIDS** | Implemented | continuous + events + channels + **electrodes.tsv** + **coordsystem.json** + **iEEG** · `load_bids_modality_full` · `write_full_bids_stub` |
| **Artifact reject rates + train gate** | Implemented | `m__artifact_reject` · Dataset cards summary · **Settings: hard neuro reject gate** (default max 30%) blocks train |
| **Trial × EEG joint tokens** | Implemented | **Stimuli / trials → Run trial×EEG pipeline** · align events to continuous → tokens · `mf/neuro/trial_eeg.py` |
| **MNE / nilearn BaseTokenizers** | Registered | `mne_raw_v1`, **`mne_epochs_v1`** (Raw+events→Epochs→TokenBatch + reject flags), `nilearn_nifti_v1` (**offline cache** or schaefer/aal/ho) |
| **Offline atlas cache** | Implemented | `mf/neuro/atlas_cache.py` · Settings “Build offline atlas cache” · env `MH_ATLAS_CACHE` |
| **Epochs → reject summary auto-wire** | Implemented | Dataset cards auto-merges MNE/trial epoch tokens into `neuro_reject_summary` |

**BIDS layout (minimal)**

```text
root/
  participants.tsv
  sub-001/ses-01/eeg/sub-001_ses-01_task-rest_eeg.tsv
  sub-001/ses-01/meg/...
  sub-001/ses-01/fnirs/...
```

Empty BIDS path in the EEG tab can write a **demo tree** for smoke tests.

**Reject QC how to read:** reject rate ≫ 20–30% → tighten acquisition or preprocessing before FM/probe claims.

**Trial×EEG path:** upload/load events + continuous → multi-subject align (or trial×EEG button) → tokens → Export / probe like other modalities.

**MNE/nilearn tokenizers:** `mne_raw_v1` loads FIF → continuous → same EEG continuous feature path; `nilearn_nifti_v1` summarizes NIfTI/ROI tables into `f__*` tokens. Both emit TokenBatch like every other registry entry.

### Explainability & product

| Capability | Status | Where |
|------------|--------|--------|
| **HTML + PDF study report** | Implemented | **Results → Generate study report (HTML + PDF)** · same content · PDF via reportlab or matplotlib · `mf/reports/generator.py` |
| **Cohort comparison** | Implemented | **Playground → split feature + percentile → Compare cohorts** · high vs low OOF accuracy/F1 |
| **Playground presets** | Implemented | +2k steps, more sedentary, better sleep, higher arousal · class probabilities + margin uncertainty |

**Report contents:** row metrics · subject-level summary · leakage/protocol · artifact path · top mean abs SHAP · one local explanation line · fixed limitations (not diagnosis, not causal).

**Playground discipline:** presets are **model counterfactuals**, not treatment effects. Use with subject-level metrics and SHAP, not as clinical advice.

### Suggested order

1. Neuro: BIDS or upload → tokenize → Dataset cards reject rates  
2. Optional: Stimuli trial×EEG  
3. Train + SHAP  
4. Results → HTML report  
5. Playground presets / cohort split for stakeholder narratives  
""",
        },
        {

            "id": "improvements_inventory",
            "title": "22. Full improvements inventory (current package)",
            "body": """
This section is the **source of truth** for what has been implemented in the studio so far.
Use it when onboarding, writing methods text, or deciding what is safe to claim.

---

### A. Data contract & QC

| Item | Status | Where |
|------|--------|--------|
| Multi-file loaders (surveys, activity, labs, notes) | Done | Data panel · column maps · validation reports |
| QC dashboard (missingness, coverage, label balance) | Done | Data panel before train |
| Single **canonical panel** in `session_state` | Done | All train / FM / stimuli tabs read it |
| Target factory (PHQ-9 / GAD-7 / energy + validation) | Done | Scoring from item columns |
| Input snapshot with every custom train | Done | `exports/runs/<run>_inputs_<stamp>/` · panel + QC JSON |
| Label prevalence alerts | Done | Rare-class warning · suggests balanced/focal |

---

### B. Evaluation discipline (highest leverage)

| Item | Status | Where |
|------|--------|--------|
| GroupKFold by subject | Done | All classic + FM probe paths |
| Nested / repeated GroupKFold + CIs | Done | Eval mode in sidebar / Settings |
| Subject-level metrics (mean AUC per held-out subject) | Done | Results · `artifacts` · Guide §8 |
| Leakage audit (no subject in train∩test) | Done | `audit_groupkfold_splits` |
| **Hard leakage gate** (refuse train) | Done | Settings · `mf/probe/gates.py` |
| **Hard QC gate** (min subjects / label fill) | Done | Settings · custom train |
| Time-aware split option (next-day targets) | Done | Eval mode `time_aware` |
| `metrics.json` + `predictions.parquet` every run | Done | `evaluate_and_save` |
| Paper-style sentence (`paper_claim`) | Done | Results + HTML/PDF report |
| Subject AUC/accuracy **95% CI** in reports | Done | HTML + PDF generators |

---

### C. Classic modeling

| Item | Status | Where |
|------|--------|--------|
| RF + XGBoost, classification/regression | Done | Custom train |
| Windowing, sliding, next-day target | Done | Feature engineering sidebar |
| Imbalance: class weights / focal-style weights | Done | Imbalance mode |
| Calibrated ensemble (RF+XGB after isotonic) | Done | Calibration tab |
| Ablation by modality groups | Done | Ablation automation |
| Optuna for XGB (+ history, contour, etc.) | Done | Optuna tab |
| Nested GroupKFold + **nested Optuna** (inner tune) | Done | `nested_optuna_groupkfold` · Results button |
| Hybrid subject×time + time-aware splits | Done | `hybrid_subject_time_cv` · `time_aware_group_split` |
| Decision curves / threshold utility | Done | `decision_curves.py` · Results |
| Ablation + **subject CIs** | Done | `run_ablation_from_xy` |
| Ordinal PHQ (QWK, MAE) | Done | Task `ordinal_phq` · `ordinal.py` |
| Nested-style discipline (outer GroupKFold) | Done | Eval protocol |

---

### D. Explainability

| Item | Status | Where |
|------|--------|--------|
| Impurity / permutation / correlation importance | Done | Results · Feature importance |
| **Highest leverage features** (combined rank) | Done | Results · impurity+corr+SHAP±perm |
| SHAP beeswarm, bar, waterfall, dependence | Done | SHAP tab |
| SHAP interaction / combined effects | Done | Interactions |
| Rank stability ideas | Done | Importance suite |

---

### E. Neuro stack

| Item | Status | Where |
|------|--------|--------|
| EEG continuous tokenization (epochs, bandpower, spectrogram) | Done | Tokenizers · EEG tab |
| Artifact QC + reject rates in Dataset cards | Done | `m__artifact_reject` |
| BIDS continuous + events/channels + **electrodes/coordsystem/iEEG** | Done | `load_bids_modality_full` · `write_full_bids_stub` |
| **Offline atlas cache** (CI / no network) | Done | `mf/neuro/atlas_cache.py` · Settings button · `MH_ATLAS_CACHE` |
| **Auto-wire MNE/trial epochs → reject summary** | Done | `_auto_wire_epoch_rejects` · Dataset cards |
| Trial×EEG joint tokens | Done | Stimuli · `trial_eeg_pipeline` |
| MEG / fNIRS / fMRI / Spikes-LFP tokenizers | Done | Registry · BIDS parity load |
| **mne_raw_v1** / **mne_epochs_v1** / **nilearn_nifti_v1** | Done | Epochs path + Schaefer/AAL/HO atlases |
| **Neuro reject-rate train gate** | Done | Settings · default 30% · `neuro_reject_gate` |
| Optional MNE FIF → continuous / epochs | Done | `mf/neuro/mne_adapter.py` |

---

### F. Stimuli & behaviour

| Item | Status | Where |
|------|--------|--------|
| Stimulus / behaviour schemas | Done | `mf/stimuli.py` |
| trial_align buffers + baseline | Done | Stimuli tab |
| Event→daily rollups | Done | For MH targets |
| Multi-subject align + PEA / raster | Done | Stimuli visualizations |

---

### G. Foundation-model path

| Item | Status | Where |
|------|--------|--------|
| Unified timeline + missingness masks | Done | FM preprocess |
| Sequence JSONL export | Done | Foundation Model tab |
| Contrastive pair strategies | Done | SSL pairs export |
| Per-subject normalization in manifest | Done | FM config |
| **Minimal PyTorch train** (Transformer/TCN) | Done | `mf/fm/*` · masked + contrastive |
| Checkpoint `W.npy` / `encoder.pt` | Done | FM train out dir |
| Auto-discover `sequences.jsonl` | Done | FM tab path field |
| Subject **holdout** during pretrain | Done | Default 20% subjects |
| Feature align probe X → checkpoint keys | Done | `mf/fm/align_features.py` |
| FM Probe: frozen encoder + linear head | Done | Same GroupKFold as trees |
| Linear probe vs RF benchmark | Done | FM tab |
| Load `.npy` / `.npz` / `.pt` into probe | Done | Checkpoint encoder |

---

### H. Product / reports

| Item | Status | Where |
|------|--------|--------|
| HTML study report | Done | Results |
| **PDF study report** (reportlab or matplotlib) | Done | Same button · download PDF |
| Cohort comparison (high vs low feature) | Done | Playground |
| Narrative presets (+2k steps, etc.) | Done | Playground · counterfactuals |
| Guide tab (this manual) | Done | Full manual + diagrams data |

---

### I. Package layout (reference)

```text
src/mf/
  align.py, labels.py, pipeline.py, schema.py, stimuli.py
  datasets/   simulate, export, manifest, fm_pretrain, stress, token_viz
  tokenize/   base, registry, activity, eeg, meg, fnirs, fmri, spikes, audio, notes, mne_nilearn
  neuro/      bids, trial_eeg, mne_adapter
  probe/      classic, metrics, artifacts, evaluation, gates, importance, imbalance, ensemble, ablation, fm_probe
  fm/         dataset, models, train, align_features
  reports/    generator (HTML+PDF)
  guide_content.py
app.py · scripts/train_fm_min.py · configs/*.yaml
```

---

### J. Suggested claim language

Safe: subject-wise GroupKFold metrics, leakage audit OK, artifact paths, highest-leverage feature *associations*, FM linear probe vs RF on held-out subjects.

Unsafe without external validation: clinical deployment, causal effect of step changes from Playground, “foundation model” performance beyond the tiny pretrain scaffold.
""",
        },

        {
            "id": "modeling_tabular",
            "title": "Modeling (tabular / probe) discipline",
            "body": """
Eval modes: groupkfold, repeated_groupkfold, nested_groupkfold, **nested_optuna**, time_aware, **hybrid_subject_time**.

Nested Optuna: outer GroupKFold metrics; inner Optuna on train subjects only.

Decision curves on OOF probabilities; ablation with subject AUC CIs; ordinal_phq with QWK.
""",
        },
        {
            "id": "fm_horizon",
            "title": "Foundation-model horizon (SSL · fusion · external)",
            "body": """
### Larger SSL
Longer max_len + optional **contrastive pairs** file → triplet phase in `train_fm_encoder` (`mf/fm/ssl_pairs.py`).

### External frozen encoder
Resolve W.npy / encoder.pt / embeddings / HF folder (`mf/fm/external_encoder.py`) → FM Probe checkpoint.

### Multimodal fusion sequences
Merge daily + EEG/trial frames → `sequences.jsonl` with modality prefixes and presence masks.

### External validation cohort
Train on current Xy; evaluate on second cohort dir; subject CIs + artifacts (`external_validation.py`).

### Still larger
Heavy multi-GPU SSL, published neuro FMs, multi-site validation packs.
""",
        },

        {
            "id": "entry_clinical_neuro",
            "title": "Entry data: clinical/behavioral + neuroscience streams",
            "body": """
### Core clinical / behavioral
Surveys (PHQ/GAD/energy/pain/substance), activity, sleep, vitals (HR/HRV/BP), labs, notes, behavior.
Canonical panel via MODALITY_COLUMN_MAPS; targets via attach_survey_labels + derived outcomes.

### Neuroscience streams
EEG/MEG/fNIRS/iEEG BIDS + trial events. Bridge into panel with stream_entry.py.
Order: clinical panel → BIDS/stimuli → Bridge neuro → tokenize / FM fusion / probe.
""",
        },

        {
            "id": "fm_entry_products",
            "title": "FM entry products (sequences · pairs · packs · W)",
            "body": """
Products: sequences.jsonl, contrastive pairs, fusion sequences, cohort packs, W bundle.
App: Foundation Model → FM entry products. Code: mf/fm/entry_products.py
""",
        },

        {
            "id": "engineering_tiers",
            "title": "Engineering tiers 1–4 (studio depth)",
            "body": """
Tier 1 Trust · Tier 2 Neuro · Tier 3 FM path · Tier 4 Product ops.
Settings → Verify engineering modules. configs/studio_experiment_v1.yaml · mf/engineering_tiers.py
""",
        },

        {
            "id": "science_workflows",
            "title": "Science workflows (six pairings)",
            "body": """Next-day mood · brain↔mood · task neuro · pretrain→probe · generalization · real-world collection. Tab Workflows. mf/workflows.py""",
        },
        {
            "id": "multi_input_pipeline",
            "title": "Multi-input intake and full analysis",
            "body": """
Load many modality CSVs into one subject–date panel, window, then run studio ML.

### Modules
- `mf.ingest.multi_input` — schemas, loaders, survey scoring, merge, QC, demo bundle
- `mf.ingest.multi_input_analysis` — windows, GroupKFold RF/XGB, paper path, importance, calibration, optional wider transfer
- CLI: `scripts/run_multi_input_analysis.py`

### Modalities
surveys, activity, vitals, labs, sleep, meds, pain, substances, screen, weather, trial_behavior, eeg_daily, notes_daily, generic

### Run demo
```bash
PYTHONPATH=src python scripts/run_multi_input_analysis.py --demo --wider --out exports/mi
```

### Run uploads
```bash
PYTHONPATH=src python scripts/run_multi_input_analysis.py \
  --surveys path/surveys.csv --activity path/activity.csv \
  --vitals path/vitals.csv --sleep path/sleep.csv \
  --label phq9_elevated --model rf
```

### Artifacts
canonical_panel.csv, analysis_summary.json, paper_path metrics/predictions, feature_importance.csv
""",
        },
        {
            "id": "tier1_leverage",
            "title": "Tier-1 highest leverage (smoke, paper path, cloud, YAML)",
            "body": """
### Why Tier-1 first
New modalities matter less than **trusted, reproducible runs**.

### 1. End-to-end smoke tests
```bash
PYTHONPATH=src python scripts/smoke_test_studio.py
PYTHONPATH=src python scripts/smoke_test_studio.py --quick
```
Checks imports, DataFrame-safe workflow flags, simulate→window, **paper path** (leakage + artifacts + claim), wider framing, YAML configs.

### 2. Locked paper path
Module: `mf.probe.paper_path` · CLI: `scripts/run_paper_path.py` · config: `configs/paper_path_v1.yaml`

Always produces:
- GroupKFold by subject + leakage audit
- metrics.json, predictions.csv, per_subject_metrics.csv
- paper_claim sentence
- run_meta.json (config + xy hash)

```bash
PYTHONPATH=src python scripts/run_paper_path.py --config configs/paper_path_v1.yaml
```

### 3. Cloud / demo profile
- `configs/cloud_demo.yaml` — demo defaults, EEG/audio/optuna off
- `requirements-cloud.txt` — lean Community Cloud deps (no torch/MNE)

### 4. YAML as source of truth
Paper and cloud configs live under `configs/`. Prefer CLI + YAML for any table that enters a manuscript; UI is for exploration.

### Definition of done
1. Smoke exit code 0 on a clean clone
2. Paper path artifacts on every publishable train
3. Demo works with zero uploads on Cloud
""",
        },
        {
            "id": "wider_dynamics_suite",
            "title": "Wider FM dynamics suite (8 axes)",
            "body": """
Eight axes beyond subject-holdout PCA (`mf.fm.wider_dynamics`):

1. **Cross-task** — train_task × test_task probe matrix
2. **Hierarchy** — fast/mid/day pooling eval
3. **Pretenses** — masked reconstruct, next-step, contrastive neighbors (sklearn stand-ins)
4. **Domain** — hold out site/device
5. **Rich eval** — few-shot curve, temporal generalization, RSA
6. **Joint brain–behavior** — concat vs neural-only vs behavior-only
7. **Longitudinal** — fit early windows, probe late
8. **Intervention** — feature bumps → Δpred and Δz

CLI: `PYTHONPATH=src python scripts/run_wider_dynamics.py --out exports/wider_dynamics`

UI: Foundation Model → *Dynamics suite (8 axes)*.

These are evaluation protocols + lightweight pretenses; full torch SSL still lives in `train_fm_scale.py`.
""",
        },
        {
            "id": "wider_fm_framing",
            "title": "Wider FM scientific framing (shift the question)",
            "body": """
### From clinical predictor to neural foundation

**Classic question:** How well do windowed features predict PHQ/GAD in this cohort?

**Wider FM question:** Do learned representations of brain/behavior dynamics **transfer** across subjects (and sites), and support **multiple heads**?

Code: `src/mf/fm/wider_framing.py` · CLI: `scripts/run_wider_fm_framing.py` · UI: Foundation Model tab → *Wider FM scientific framing*.

### Declared scientific questions

| ID | Kind | Claim |
|----|------|--------|
| `in_cohort_mood` | clinical_predictor | Within-cohort mood prediction (classic path) |
| `representation_transfer` | neural_foundation | Fit representation on subject pool A; linear probe only on held-out pool B |
| `multi_head_dynamics` | joint_brain_behavior | One frozen z supports several heads (mood + behavior aux, …) |
| `cross_site_foundation` | neural_foundation | Same as transfer, with site holdout when site labels exist |

### Experiment protocol (no probe leakage)

1. Split subjects into **fit_subjects** vs **probe_subjects** (fraction or site).
2. Fit representation (**PCA / identity / standardize**) **only** on fit-pool rows.
3. Transform all rows → Z.
4. For each head: GroupKFold **linear probe within probe pool**; optional RF on raw X for the same rows; fit-pool linear as in-distribution reference.
5. Save `wider_fm_framing_result.json` + summary table.

### How to run

```bash
PYTHONPATH=src python scripts/run_wider_fm_framing.py \
  --question representation_transfer \
  --representation pca \
  --heldout-fraction 0.3 \
  --out exports/wider_fm_demo
```

Or: Custom train → Foundation Model → expander → **Run wider framing experiment**.

### Reading results

| Pattern | Reading |
|---------|---------|
| Probe linear ≈ fit linear | Representation transfers across subjects |
| Probe linear ≪ RF | Trees still win on raw features; representation weak/linear |
| Probe linear ≈ RF | Strong transferable representation for that head |
| Several heads OK | Shared dynamics, not a single-label artifact |

### Relation to existing FM Probe tab

| FM Probe | Wider framing |
|----------|----------------|
| Encoder + linear head under GroupKFold on all subjects | Explicit **fit vs probe subject pools** |
| One primary label | **Multi-head** specs |
| Checkpoint W from SSL | PCA/identity stand-in until SSL W is plugged into the same transfer protocol |

Next: plug SSL `W.npy` into `fit_representation` / checkpoint path using the same transfer split API.
""",
        },
        {
            "id": "gaps",
            "title": "23. Implemented vs frontiers",
            "body": """Six science workflows + four engineering tiers in studio.

Also: wider FM framing (`wider_fm_framing`) — transfer subject pools, multi-head linear probes, representation vs RF.
""",
        },
    ]




NEUROSYMBOLIC_SECTION = {
    "id": "neurosymbolic",
    "title": "Neurosymbolic reasoning (neural scores + clinical rules)",
    "body": """
### Why neurosymbolic here?

The studio already trains **statistical / neural-style probes** (RF, XGBoost, linear heads on representations).
Clinicians and protocols also use **explicit definitions**: PHQ-9 ≥ 10, short sleep, high sedentary time.

**Neurosymbolic integration** keeps both:

| Layer | Role |
|-------|------|
| Neural / ML | Learns P(elevated mood | features / tokens) |
| Symbolic | Predicates + if–then rules + severity bands |
| Hybrid | `(1-α)·P_model + α·symbolic_score` |
| Trace | Which rules fired — auditable narrative |

### Studio procedure

1. Build **canonical panel** (Data panel — all modalities)  
2. Optional: **Custom train** so `model` / OOF proba exist  
3. Open **Foundation Model → Neurosymbolic** (or run `mf.neurosymbolic`)  
4. Inspect predicate rates, rule fire counts, hybrid scores, force-review flags  
5. Tune **α** (0 = pure model, 1 = pure rules)

### Default rule examples

- High sedentary **and** low steps → boost risk context  
- Sleep duration proxy low → boost  
- PHQ-9 ≥ 10 → **force_review** (screening cutoff)  
- Not low-activity → slight penalize of elevated-risk score  

### Limits (read carefully)

- Rules are **research defaults**, not practice guidelines  
- Hybrid scores are **not diagnoses**  
- Always report **GroupKFold / subject-holdout metrics** separately from hybrid overlays  
- Symbolic features need columns present (demo multi-input provides them)

### Link to FM evaluation culture

Neurosymbolic rules are **orthogonal** to FM vs classic ML culture:

- Classic MH: rules on top of RF OOF proba  
- Neuro FM: rules on top of linear-probe proba from frozen encoder  
- Either way, leakage discipline still applies to the **learned** part
""",
}


def guide_sections_with_neurosymbolic() -> list[dict[str, str]]:
    """guide_sections() plus neurosymbolic chapter."""
    secs = guide_sections()
    ids = {s.get("id") for s in secs}
    if "neurosymbolic" not in ids:
        secs.append(NEUROSYMBOLIC_SECTION)
    return secs



RESEARCH_MODULES_SECTION = {
    "id": "research_modules_forks",
    "title": "Research modules · conceptual forks · fresh prospectives",
    "body": """
### Product / research modules

Named packages with a fixed scientific question and runner:

| Module | Question (short) |
|--------|------------------|
| next_day_mood | Prior features → next-day target |
| brain_behavior_link | Does eeg_daily add beyond activity/vitals? |
| task_neuro_bridge | Trial RT/acc features vs daily-only |
| pretrain_probe_lab | Frozen linear probe vs RF same splits |
| generalization_pack | Held-out subjects + leakage audit |
| real_world_qc | Coverage / label gates before train |
| neurosymbolic_overlay | Rules + hybrid on model proba |
| dual_eval_culture | Classic MH vs neuro-FM metrics |
| intervention_proxy | +steps / −sedentary delta-P |
| uncertainty_narrative | Research text with confidence language |

App: **Foundation Model → Research modules**. Code: `mf.research_modules`.

### Conceptual forks (point 6)

| Fork | Pole A | Pole B |
|------|--------|--------|
| Predict vs represent | Downstream y metrics | Pretext / RSA / transfer |
| In-cohort vs holdout | GroupKFold | Subject holdout |
| Flat vs sequence | Tabular f__ | JSONL tokens |
| End-to-end vs frozen | RF/XGB on X | Linear probe on Z |
| Pure neural vs neurosym | Proba only | Hybrid rules |
| Utility vs validity | Narratives / thresholds | Leakage-audited transfer |

Run **Compare forks on current Xy** for an empirical A/B table.

### Fresh prospectives (point 7)

Stack-native extensions (no new sensor types required):

1. Counterfactual activity narratives (intervention proxy)
2. Temporal stability of predictions within subject
3. Synthetic multi-site leave-one-site-out
4. Modality necessity ablation grid
5. Probe mechanics (PCA loadings × linear coef)
6. Narrative cards from module outputs
7. Missingness rate as an explicit feature
8. Contrastive geometry margins by subject

Code: `mf.fresh_prospectives` · App button **Run fresh prospective suite**.
""",
}



def guide_sections_full() -> list[dict[str, str]]:
    secs = guide_sections_with_neurosymbolic()
    if not any(s.get("id") == "research_modules_forks" for s in secs):
        secs.append(RESEARCH_MODULES_SECTION)
    return secs


GUIDE_SECTIONS = guide_sections_full()




def architecture_sankey_data() -> dict[str, Any]:
    labels = [
        "Raw modalities",
        "Alignment",
        "TokenBatch",
        "Flat table",
        "Sequences",
        "RF/XGB",
        "Linear probe",
        "SHAP",
        "FM pretrain (ext)",
    ]
    links = [
        (0, 1, 8),
        (1, 2, 8),
        (2, 3, 5),
        (2, 4, 3),
        (3, 5, 4),
        (3, 6, 2),
        (5, 7, 3),
        (6, 7, 1),
        (4, 8, 3),
    ]
    return {
        "labels": labels,
        "source": [a for a, b, v in links],
        "target": [b for a, b, v in links],
        "value": [v for a, b, v in links],
    }


def split_diagram_levels() -> list[dict[str, Any]]:
    return [
        {"fold": "Fold 0", "train": "S1 S2 S3 S4", "test": "S5"},
        {"fold": "Fold 1", "train": "S1 S2 S3 S5", "test": "S4"},
        {"fold": "Fold 2", "train": "S1 S2 S4 S5", "test": "S3"},
        {"fold": "…", "train": "…", "test": "held-out subjects"},
    ]
