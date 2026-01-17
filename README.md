# Sim2Real-Biometrics Assessment

Bridging the Domain Gap between Synthetic Training and Real-World Facial Recognition.

## 🚀 Overview

This project addresses the **Sim2Real** challenge in biometrics: training models on high-fidelity synthetic data (**DigiFace**) and evaluating their performance on diverse, real-world datasets (**LFW**).

We explore how aggressive data augmentation (blur, noise, perspectives) and architectural choices (ResNet-50 vs ResNet-18) impact the model's ability to generalize to real human faces.

## ✨ Key Features

- **Face Alignment**: Automated MTCNN-based alignment for both training and testing domains.
- **Backbone Models**:
  - **Proposed (Strong)**: An optimized ResNet-50 with strong Sim2Real augmentations (Blur, Perspective) and Cosine Annealing.
  - **Baseline (Standard)**: Standard ResNet-50 trained with basic DigiFace augmentations.
  - **Scratch (Random)**: ResNet-18 trained from random initialization without synthetic pre-training to measure the "transfer gap".
- **ArcFace Loss**: Metric learning for highly discriminative feature embeddings.
- **Interpretability**: Integrated **Grad-CAM** to visualize where the model focuses (Texture vs. Shape).
- **Live Demo**: A real-time webcam security system simulation to test enrollment and verification.

## 📊 Benchmark Results (on LFW)

| Model | AUC (Stability) | EER (Equilibrium) | TAR @ 1e-3 FAR |
| :--- | :--- | :--- | :--- |
| **Proposed (Strong)** | **0.911** | **16.03%** | **25.22%** |
| **Baseline (Standard)** | 0.871 | 21.08% | 12.39% |
| **Scratch (Random)** | 0.814 | 28.52% | 4.88% |

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/codesob/sim2real-biometrics.git
cd sim2real-biometrics

# Install dependencies
pip install -r requirements.txt
```

## 📈 Training and Evaluation

### 1. Preprocessing

Align the raw DigiFace images to the canonical 112x112 format:

```bash
python preprocess.py
```

### 2. Training

We provide two training paths:

- **Baseline**: `python train_baseline.py` (Standard augmentations).
- **Proposed (Scratch)**: `python train_scratch.py` (Strong Sim2Real augmentations + Cosine Annealing).

### 3. Evaluation

Benchmark against the LFW dataset (Real-world data):

```bash
python test.py
```

This generates ROC curves, t-SNE visualizations, and biometric metrics (AUC, Accuracy).

## 🛡️ Live Security Demo

Test the trained model in real-time:

```bash
python live_demo.py
```

- Press **'E'** to enroll your face.
- The system will then perform continuous verification against your enrolled identity.

## 📊 Visual Analysis

The project includes tools to visualize the "Generalization Gap":

- `comparative_visuals.py`: Compare performance across different training stages.
- `visualize_heatmap.py`: Use Grad-CAM to see if the model is biased toward synthetic textures or representative facial shapes.

## 📄 License

This project is open-source and available under the MIT License.
