# CEG3004 DSP Mini-Project: Environmental Sound Classification

**Group:** Pr_26

## Overview

This project classifies environmental sounds from the ESC-50 dataset (50 classes, 2000 clips) using a machine learning pipeline built on scikit-learn. The approach focuses on robust audio preprocessing and rich feature extraction to handle clean, noisy, and bandlimited audio conditions.

---

## Methodology

### Preprocessing

Audio is preprocessed in 4 steps before any feature extraction:

1. **Silence trimming** — `librosa.effects.trim(top_db=25)` removes leading/trailing silence and low-amplitude sections.
2. **High-pass Butterworth filter** — 4th-order Butterworth filter at 100 Hz cutoff, applied with zero-phase (`filtfilt`) to remove low-frequency hum without phase distortion.
3. **Peak normalization** — Divides by the maximum absolute amplitude to scale all clips to the range [-1, 1].
4. **Fixed-length padding/truncation** — All clips are standardised to exactly 5 seconds at 16 kHz (80,000 samples).

> **Design rationale:** A high-pass filter was chosen over a pre-emphasis filter because the ESC-50 dataset spans sounds across the full frequency spectrum (thunderstorm vs. bird chirp). Removing only DC/sub-bass noise (<100 Hz) preserves low-frequency features that a pre-emphasis filter would suppress.

---

### Feature Extraction

Features are extracted in four groups, producing a 243-dimensional feature vector per clip:

| Group | Features | Pooling |
|---|---|---|
| MFCC (n=40) + CMVN | 40 coefficients, delta, delta-delta | mean, std, max, median (robust) |
| Log-Mel Spectrogram (n_mels=64) | Log-power mel bands | mean, std |
| Spectral Features | Centroid, bandwidth, rolloff, onset flux, ZCR | mean |

**CMVN (Cepstral Mean Variance Normalization)** is applied to MFCCs before pooling, making the features robust to channel noise and recording conditions — important for the noisy/bandlimited test sets.

**Robust pooling** (mean + std + max + median) captures more of the temporal structure than mean pooling alone.

---

### Data Augmentation

Each training clip is augmented 3x before training, giving 4x the original training data:

- **Additive Gaussian noise** — Simulates microphone noise (`σ = 0.005`)
- **Time shift** — Shifts the signal by 0.5 seconds (`np.roll`)
- **Pitch shift** — Shifts pitch up by 2 semitones (`librosa.effects.pitch_shift`)

---

### Model

Multiple models were evaluated before settling on the final choice. See the Experiment Log for results.

#### Final model: HistGradientBoostingClassifier

**Pipeline:**

```text
RobustScaler → HistGradientBoostingClassifier(max_iter=300, lr=0.05, max_depth=6, class_weight='balanced')
```

#### Models evaluated (in order)

| Model | Pipeline | Outcome |
| --- | --- | --- |
| Stacking Ensemble (ET + XGB + SVC) | RobustScaler → PCA → SMOTE → StackingClassifier | Severe class prediction bias; `rain` predicted 134/1200 times. Training too slow for iteration on Colab. |
| XGBClassifier (standalone) | RobustScaler → XGBClassifier | Faster than stacking but still showed bias toward dominant-sounding classes. |
| HistGradientBoostingClassifier | RobustScaler → HistGradientBoostingClassifier | Best balance of speed and accuracy. Handles non-linear boundaries across 50 classes with built-in class balancing. |

HistGradientBoosting was selected as it handles non-linear decision boundaries natively, trains significantly faster than the stacking ensemble, and uses `class_weight='balanced'` to correct for class imbalance without requiring SMOTE.

---

## Comparison with Reference Approach

The table below compares our approach against a reference implementation (Pr_16) also based on the same CEG3004 template:

| Aspect | Our Approach (Pr_26) | Reference (Pr_16) |
|---|---|---|
| Preprocessing | Silence trim + Butterworth HPF + peak norm | Silence trim + pre-emphasis + RMS norm |
| Filter type | High-pass (removes <100 Hz) | Pre-emphasis (boosts high freq) |
| Audio duration | 5 seconds | 3.5 seconds |
| MFCC coefficients | 40 | 40 |
| MFCC normalization | CMVN (per-coefficient) | None |
| Mel bands | 64 | 255 |
| Delta features | delta + delta-delta | delta + delta-delta |
| Chroma features | No | Yes |
| Robust pooling | Yes (mean/std/max/median) | Stats only |
| Data augmentation | Yes (noise, shift, pitch) | No |
| Model | Stacking (ET + XGB + SVC) | LogisticRegression |
| Class imbalance handling | SMOTE | Balanced class weights |
| Feature vector size | 243 | Larger (255 mel bands + chroma) |

---

## Experiment Log

| Run | Preprocessing | Feature Extraction | Model | Notes |
|---|---|---|---|---|
| Baseline | None | Default MFCC | LogisticRegression | Template default |
| v1 | Trim + HPF + peak norm | MFCC-40 + deltas | RandomForest | Wrong run order |
| v2 | Trim + HPF + peak norm | MFCC-40 + deltas + log-mel + spectral | Stacking Ensemble | Correct run order, augmented |

---

## Steps to Reproduce

1. Open `source_code` (Google Colab notebook) in Google Colab.
2. Mount Google Drive and set the correct paths to the ESC-50 dataset.
3. Run cells **in order** — do **not** use "Run All" (see note below).
4. Run the augmentation cell **before** the model training cell.
5. After training, save `Pr_26_model.joblib` and generate `Pr_26_predictions.csv`.

### Cell Run Order

> **Why you cannot use "Run All":**
> The notebook has two separate cells that build the training dataset:
> - **Cell 15** — builds `X, y` from the original (non-augmented) training data.
> - **Cell 19** — rebuilds `X, y` with augmented data (4x size).
>
> If you Run All, Cell 15 creates `X, y`, then the model cell runs and trains on the small non-augmented dataset. Cell 19 then overwrites `X, y` but training already happened — so augmentation has no effect.

Run cells in this order:

| Step | Cell | What it does |
|---|---|---|
| 1 | Setup cells | Install libraries, imports |
| 2 | Data loading | Download ESC-50, load metadata |
| 3 | Cell 13 | Define `preprocess_audio` |
| 4 | Cell 14 | Define `extract_features` |
| 5 | **Cell 19** | Build augmented `X, y` (run this BEFORE model definition) |
| 6 | Train/val split | Split into train and validation |
| 7 | **Cell 17** | Define model |
| 8 | Training cell | `model.fit(X_train, y_train)` |
| 9 | Evaluation | Print accuracy and F1 score |
| 10 | Cell 22 | Predict on test set, save CSV |
| 11 | Save model | `joblib.dump(model, ...)` |

---

## Dependencies

```
librosa
scikit-learn
xgboost
imbalanced-learn
scipy
numpy
pandas
tqdm
joblib
soundfile
```

Install in Colab:
```bash
pip install librosa xgboost imbalanced-learn soundfile
```

---

## Repository Structure

```
.
├── README.md
├── source_code              # Main Colab notebook
├── Pr_26_model.joblib       # Trained model
└── Pr_26_predictions.csv    # Test set predictions
```
