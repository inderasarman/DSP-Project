# CEG3004 DSP Mini-Project: Environmental Sound Classification

**Group:** Pr_26

## Overview

This project designs a robust audio classification pipeline for Environmental Sound Classification (ESC-50) that performs well under clean, noisy, and bandlimited conditions. The pipeline covers four stages: audio preprocessing, DSP feature extraction, data augmentation, and machine learning classification across 50 sound classes.

---

## Results

| Split | Accuracy | Macro-F1 |
|---|---|---|
| Validation (20% holdout) | **0.87** | **0.886** |

The validation set uses the same clean audio distribution as the training set. The submission set additionally tests robustness under additive noise and bandlimited conditions.

---

## Methodology

### 1. Preprocessing

Audio is preprocessed in 4 steps before feature extraction:

1. **Silence trimming** — `librosa.effects.trim(top_db=25)` removes leading/trailing silence. A threshold of 25 dB was chosen to aggressively remove near-silence without clipping soft-onset sounds like distant thunder or rustling leaves.
2. **High-pass Butterworth filter** — A 4th-order Butterworth filter with a 100 Hz cutoff, applied zero-phase using `scipy.signal.filtfilt` to remove low-frequency hum and DC offset without introducing phase distortion into the signal.
3. **Peak normalization** — Divides by the maximum absolute amplitude to scale all clips to [-1, 1], making feature magnitudes independent of recording gain.
4. **Fixed-length padding/truncation** — All clips are standardised to exactly 5 seconds at 16 kHz (80,000 samples), ensuring a consistent feature vector size regardless of original clip length.

> **Design rationale — HPF vs. pre-emphasis:**
> A high-pass filter was chosen over a pre-emphasis filter because ESC-50 spans sounds across the full frequency spectrum (e.g., thunderstorm energy is concentrated below 500 Hz; bird chirps are above 2 kHz). A pre-emphasis filter (e.g., `y[n] - 0.97*y[n-1]`) boosts all high frequencies globally, which would suppress low-frequency discriminative features for classes like thunderstorm, rain, and engine idling. The HPF at 100 Hz instead removes only sub-bass noise and DC drift, leaving the full useful spectrum intact.

> **Design rationale — zero-phase filtering:**
> `filtfilt` applies the filter forwards and backwards, resulting in zero phase distortion. This preserves the temporal alignment of features (onset, attack), which matters for transient sounds like dog barks and gun shots.

---

### 2. Feature Extraction

Features are extracted in four groups, producing a **243-dimensional feature vector** per clip:

| Group | Features | Pooling | Dim |
|---|---|---|---|
| MFCC (n=40) + CMVN | 40 coefficients + delta + delta-delta | mean, std, max, median | 480 → see note |
| Log-Mel Spectrogram (n_mels=64) | Log-power mel filterbank | mean, std | 128 |
| Spectral Features | Centroid, bandwidth, rolloff, onset flux, ZCR | mean | 5 |
| — | Total | — | **243** |

**CMVN (Cepstral Mean and Variance Normalization):** Applied per-coefficient across time frames before pooling. CMVN subtracts the per-frame mean and divides by the standard deviation, making MFCC features robust to channel effects and recording-level differences — directly addressing the noisy and bandlimited test conditions.

**Robust pooling (mean + std + max + median):** Mean alone cannot capture transient events (e.g., a single dog bark in a 5-second clip). Adding std, max, and median ensures the feature vector encodes both the average spectral shape and the temporal dynamics of the sound.

**Log-Mel Spectrogram:** The mel scale approximates human auditory perception by spacing frequency bands logarithmically. Log compression (`librosa.power_to_db`) further compresses the dynamic range of the spectrogram, making features less sensitive to absolute energy levels — useful under the noisy test condition where additive noise raises the noise floor.

**Spectral features:**
- *Centroid* — the "centre of mass" of the spectrum; high for bright/hissy sounds (chainsaw), low for rumbling sounds (thunder).
- *Bandwidth* — spread around the centroid; wide for noise-like sounds, narrow for tonal sounds.
- *Rolloff* — the frequency below which 85% of energy lies; distinguishes high-energy broadband sounds from narrow-band sounds.
- *Onset flux* — measures sudden energy increases; captures the attack envelope of percussive or transient sounds.
- *ZCR (Zero-Crossing Rate)* — a time-domain feature that is high for noise-like signals and low for tonal signals. Importantly, ZCR is relatively unaffected by bandlimiting since it is computed from the raw waveform, providing robustness to the bandlimited test condition.

---

### 3. Data Augmentation

Each training clip is augmented 3 times before training, producing **4× the original training data (8,000 samples)**:

| Augmentation | Implementation | Rationale |
|---|---|---|
| Additive Gaussian noise | `y + 0.005 * randn(len(y))` | Simulates microphone noise; trains model to ignore low-level noise — directly targets the noisy test condition |
| Time shift | `np.roll(y, int(sr * 0.5))` | Shifts signal by 0.5 s; teaches the model that sound class labels are position-invariant |
| Pitch shift | `librosa.effects.pitch_shift(y, n_steps=2)` | Shifts pitch by +2 semitones; increases intra-class variation to reduce overfitting |

---

### 4. Model

#### Final model: HistGradientBoostingClassifier

```text
RobustScaler → HistGradientBoostingClassifier(max_iter=300, lr=0.05, max_depth=6, class_weight='balanced')
```

**RobustScaler** was chosen over StandardScaler because it uses the median and interquartile range rather than mean and variance, making it less sensitive to outlier feature values that can arise from clipped or very short audio clips.

**HistGradientBoostingClassifier** was selected after testing four model types (see Experiment Log). Key reasons:
- Natively handles non-linear decision boundaries across 50 classes without kernel tricks.
- Built-in `class_weight='balanced'` corrects for class imbalance without requiring SMOTE oversampling.
- Trains significantly faster than the stacking ensemble tested in Run 3, enabling more iterations within Colab's session time limit.
- `max_depth=6` provides enough capacity for 50-class separation while avoiding overfitting on the augmented dataset.

---

## Comparison with Reference Approach

| Aspect | Our Approach (Pr_26) | Reference (Pr_16) |
|---|---|---|
| Preprocessing | Silence trim + Butterworth HPF + peak norm | Silence trim + pre-emphasis + RMS norm |
| Filter type | High-pass (removes <100 Hz) | Pre-emphasis (boosts high freq) |
| Audio duration | 5 seconds | 3.5 seconds |
| MFCC coefficients | 40 | 40 |
| MFCC normalization | CMVN (per-coefficient) | CMVN |
| Mel bands | 64 | 255 |
| Delta features | delta + delta-delta | delta + delta-delta |
| Robust pooling | Yes (mean/std/max/median) | Yes (7 stats) |
| Data augmentation | Yes (noise, shift, pitch) | No |
| Model | HistGradientBoostingClassifier | LogisticRegression |
| Class imbalance handling | class_weight='balanced' | class_weight='balanced' |
| Validation accuracy | **0.87** | 0.65 |
| Macro-F1 | **0.886** | — |

---

## Experiment Log

| Run | Preprocessing | Feature Extraction | Model | Augmentation | Val Accuracy | Macro-F1 | Notes |
|---|---|---|---|---|---|---|---|
| Baseline | None | Default MFCC-20 + deltas | LogisticRegression | No | ~0.40 | — | Template default |
| v1 | Trim + HPF + peak norm | MFCC-40 + deltas | RandomForest | No | — | — | Incorrect cell run order; augmentation cell ran after training |
| v2 | Trim + HPF + peak norm | MFCC-40 + deltas + log-mel + spectral | Stacking (ET + XGB + SVC) | Yes (3×) | — | — | Severe class bias (`rain` predicted 134/1200 times); too slow for Colab |
| v3 | Trim + HPF + peak norm | MFCC-40 + deltas + log-mel + spectral | HistGradientBoostingClassifier | Yes (3×) | **0.87** | **0.886** | Final model; correct cell run order |

**Key observations:**
- The jump from Baseline to v3 demonstrates that CMVN, robust pooling, augmentation, and a non-linear classifier collectively produce the largest gains.
- The Stacking Ensemble (v2) showed class prediction bias, likely due to SMOTE oversampling on a highly skewed post-PCA feature space. Removing SMOTE and switching to `class_weight='balanced'` in HistGradient resolved this.
- Augmentation was the single biggest contributor: rerunning v3 without augmentation (correct cell order, same model) dropped Macro-F1 to approximately 0.72.

---

## Steps to Reproduce

### Requirements

```bash
pip install librosa scikit-learn scipy numpy pandas tqdm joblib soundfile
```

### Instructions

1. Open `CEG3004_Project_Colab_inderav4.ipynb` in Google Colab.
2. Download the ESC-50 dataset ZIP from the link in the notebook and place it at the path specified in Cell 3.
3. Run cells **in the order listed below** — do **not** use Runtime → Run All (see note).
4. After training completes, `Pr_26_model.joblib` and `Pr_26_predictions.csv` are auto-downloaded.

### Cell Run Order

> **Important — why Run All does not work:**
> The notebook contains two dataset-building cells:
> - **Cell 15** builds `X, y` from original (non-augmented) training data.
> - **Cell 19** rebuilds `X, y` with 3× augmented data.
>
> Running all cells sequentially causes the model (Cell 17) to train on the small non-augmented `X, y` from Cell 15. Cell 19 then overwrites `X, y` but training has already happened. Run Cell 19 **before** Cell 17.

| Step | Cell | What it does |
|---|---|---|
| 1 | Cells 1–2 | Install libraries |
| 2 | Cells 3–6 | Download and extract dataset, set paths |
| 3 | Cells 7–10 | Safety check, imports, load training labels |
| 4 | Cell 13 | Define `preprocess_audio` |
| 5 | Cell 14 | Define `extract_features` |
| 6 | **Cell 19** | Build augmented `X, y` — **run before Cell 17** |
| 7 | **Cell 17** | Train model on augmented `X, y`, save `Pr_26_model.joblib` |
| 8 | Cells 20–22 | Load submission metadata, generate and save `Pr_26_predictions.csv` |

---

## Repository Structure

```
.
├── README.md
├── CEG3004_Project_Colab_inderav4.ipynb   # Main Colab notebook (source code)
├── Pr_26_model.joblib                      # Trained model file
└── Pr_26_predictions.csv                   # Submission predictions (1200 rows)
```
