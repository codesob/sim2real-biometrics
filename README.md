# Sim2Real-Biometrics Assessment

Bridging the Domain Gap between Synthetic Training and Real-World Facial Recognition.

## 🚀 Overview

This project addresses the **Sim2Real** challenge in biometrics: training models on high-fidelity synthetic data (**DigiFace**) and evaluating their performance on diverse, real-world datasets (**LFW**).

We explore how aggressive data augmentation (blur, noise, perspectives) and architectural choices (ResNet-50 vs ResNet-18) impact the model's ability to generalize to real human faces.

## ✨ Key Features

- **Face Alignment**: Automated MTCNN-based alignment for both training and testing domains.
- **Backbone Models**: Standard ResNet-50 baseline and an optimized-from-scratch ResNet-18.
- **ArcFace Loss**: Metric learning for highly discriminative feature embeddings.
- **Interpretability**: Integrated **Grad-CAM** to visualize where the model focuses (Texture vs. Shape).
- **Live Demo**: A real-time webcam security system simulation to test enrollment and verification.

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

We provide three training paths:

- **Baseline**: `python train_baseline.py` (Standard augmentations).
- **Proposed Model**: `python train.py` (Aggressive Sim2Real augmentations: Gaussian Blur, Color Jitter, Random Perspective, Random Erasing with Cosine Annealing).
  - Saves best model to `saved_models/best_model.pth`
  - Designed to bridge the domain gap between synthetic (DigiFace) and real (LFW) data
- **Scratch Model**: `python train_scratch.py` (ResNet-18 trained from scratch with same aggressive augmentations).
  - Saves best model to `saved_models/best_scratch_model.pth`

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

## � Project Structure

├── dataset/
│   ├── digiface_aligned/        # Preprocessed DigiFace dataset (synthetic)
│   └── lfw_aligned/             # LFW dataset (real-world, for testing)
├── saved_models/                # Trained model checkpoints
│   ├── baseline_model.pth
│   ├── best_model.pth
│   ├── best_scratch_model.pth
│   └── final_model.pth
├── preprocess.py                # Dataset alignment preprocessing
├── train.py                      # Main training script
├── train_baseline.py            # Baseline training (standard augmentations)
├── train_scratch.py             # Proposed training (Sim2Real augmentations)
├── test.py                      # Evaluation on LFW
├── live_demo.py                 # Real-time webcam demo
├── visualize_heatmap.py         # Grad-CAM visualization
├── compare.py                   # Comparative analysis
├── model.py                     # Model architecture definitions
├── eda.py                       # Exploratory data analysis
└── requirements.txt             # Python dependencies

## 🔍 Key Experiments

### Baseline Model

- Standard ResNet-50 with minimal augmentations
- Reference performance on synthetic-to-real generalization
- Saves to `saved_models/baseline_model.pth`

### Proposed Model

- ResNet-50 pre-trained backbone with ArcFace loss
- **Aggressive Sim2Real Augmentations**:
  - Gaussian Blur (kernel 5-9, sigma 0.1-2.0)
  - Color Jitter (brightness, contrast, saturation: 0.2)
  - Random Perspective (distortion scale 0.2)
  - Random Erasing (probability 0.3)
  - Random Horizontal Flip
- **Learning**: Cosine Annealing scheduler with SGD optimizer
- **Goal**: Bridge synthetic-to-real domain gap
- Saves best model to `saved_models/best_model.pth`
- Train with: `python train.py`

### Scratch Model

- ResNet-18 trained from scratch (no pre-trained weights)
- Same aggressive augmentations as proposed model
- **Higher learning rate** (0.1 vs 0.01) for from-scratch training
- Explores learning directly from raw features
- Saves best model to `saved_models/best_scratch_model.pth`
- Train with: `python train_scratch.py`

## 📊 Results

Model performance metrics are saved in:

- `eda_results/` - Exploratory data analysis plots
- `final_comparison_plots/` - Performance comparison visualizations

Key metrics evaluated:

- **ROC-AUC**: Area under the ROC curve
- **True Positive Rate (TPR)** at various False Positive Rates
- **t-SNE Embeddings**: Feature space visualization
- **Grad-CAM Heatmaps**: Model interpretability analysis

## 💡 Usage Examples

### Preprocess New Data

```bash
python preprocess.py --input-dir path/to/raw/images --output-dir path/to/aligned/images
```

### Train Custom Model

```bash
python train_scratch.py --epochs 100 --batch-size 64 --learning-rate 0.01
```

### Test Model Performance

```bash
python test.py --model-path saved_models/best_model.pth --test-dir dataset/lfw_aligned/
```

### Generate Visualizations

```bash
python visualize_heatmap.py --model-path saved_models/best_model.pth --image-path sample.jpg
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## 📄 License

This project is open-source and available under the MIT License.

## 📞 Contact

For questions or collaboration inquiries, please reach out to the project maintainers.
