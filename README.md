# 🧠 EEG Motor Imagery BCI — CSP-LDA & Riemannian Geometry

Offline Brain-Computer Interface system for decoding left/right hand motor imagery from EEG signals. Implements and compares two decoding pipelines — a classical **CSP-LDA** spatial-spectral approach and an advanced **Riemannian Geometry** classifier — evaluated on the BCI Competition IV Dataset 2a.

> Developed as part of a Bachelor's Thesis in Computer Engineering at the British University in Egypt, focused on BCI-assisted neurorehabilitation.

---

## 📌 Overview

This project decodes silent motor intentions directly from raw EEG brain signals, with the goal of enabling assistive robotic control for individuals with motor disabilities. By bypassing the peripheral nervous system entirely, the system translates cortical activity into actionable digital commands using two mathematically distinct pipelines.

---

## ⚙️ Pipelines

### Model A — CSP-LDA (Spatial-Spectral Pipeline)
- Band-pass filtered at **8–30 Hz** to isolate Mu and Beta sensorimotor rhythms
- **Common Spatial Patterns (CSP)** to maximise variance contrast between left and right classes
- **Log-variance** feature extraction for Gaussian-compatible feature vectors
- **Linear Discriminant Analysis (LDA)** for binary classification

### Model B — Riemannian Geometry Pipeline
- Broader band-pass filter at **7–35 Hz** to preserve full covariance structure
- Per-trial **covariance matrix** estimation across all 22 EEG channels
- **Tangent Space Mapping (TSM)** to project SPD matrices from a curved Riemannian manifold onto a locally Euclidean space
- Inherently robust to EEG amplitude non-stationarity and baseline drift

---

## 📊 Results

Evaluated on **144 balanced trials** (72 Left, 72 Right) using **5-Fold Cross-Validation**:

| Metric | Model A (CSP-LDA) | Model B (Riemannian) |
|---|---|---|
| Accuracy | 86.11% | **97.92%** |
| CV Variance | ±1.68% | **±0.85%** |
| Right-Class Bias | 18 false positives | Negligible |

Model B achieves an **11.81 percentage-point accuracy improvement** and significantly lower variance, indicating superior generalisation and stability.

---

## 🗂️ Dataset

**BCI Competition IV — Dataset 2a**
- Subject: Subject 1 (`A01T.gdf`)
- 22 EEG channels at **250 Hz** sampling rate (International 10-20 system)
- EOG channels excluded at loading stage
- Classes: Left Hand (Class 1) vs. Right Hand (Class 2)
- Epoch window: **+0.5s post-cue** to exclude Visual Evoked Potential (VEP)

> Dataset available at: https://www.bbci.de/competition/iv/

---

## 🛠️ Requirements

```bash
pip install numpy scipy mne pyriemann scikit-learn matplotlib
```

| Library | Purpose |
|---|---|
| `MNE` | EEG data loading and preprocessing |
| `pyriemann` | Riemannian geometry and covariance estimation |
| `scikit-learn` | CSP, LDA, and cross-validation |
| `matplotlib` | Spectral and confusion matrix visualisation |

---

## 🚀 Usage

```bash
# Clone the repository
git clone https://github.com/your-username/eeg-bci-motor-imagery.git
cd eeg-bci-motor-imagery

# Run Model A (CSP-LDA)
python model_a_csp_lda.py

# Run Model B (Riemannian)
python model_b_riemannian.py
```

Place the `A01T.gdf` dataset file in the `/data` directory before running.

---

## 📁 Project Structure

```
eeg-bci-motor-imagery/
│
├── data/                   # BCI Competition IV Dataset 2a (.gdf files)
├── model_a_csp_lda.py      # CSP-LDA pipeline
├── model_b_riemannian.py   # Riemannian Geometry pipeline
├── preprocessing.py        # Shared filtering and epoching utilities
├── validation/             # Cross-validation and metrics scripts
├── results/                # Confusion matrices and spectral plots
└── README.md
```

---

## 📈 Validation

Both models were evaluated using:
- **5-Fold Cross-Validation** to prevent chronological overfitting
- **Confusion matrices** to assess directional classification bias
- **Spectral analysis** to confirm Alpha/Beta ERD as the true decoding basis
- **F1-Score** for imbalance-resistant performance measurement

---

## 🔬 Key Findings

- Riemannian geometry is significantly more robust to EEG non-stationarity than linear spatial filtering
- CSP-LDA shows a systematic right-class bias (18 false positives) absent in Model B
- Spectral validation confirms both models decode genuine cortical motor activity, not noise or artefacts
- Model B's higher computational cost is a trade-off to be addressed in future real-time implementations

---

## 🔭 Future Work

- Deep learning integration (SPDNet, Multi-Scale CNNs, Extreme Learning Machines)
- Hybrid EEG + fNIRS multimodal pipeline for artefact robustness
- Real-time closed-loop deployment in a Virtual Reality simulation environment
- Cross-subject generalisation and transfer learning

---

## 📄 License

This project is released under the MIT License. See `LICENSE` for details.
