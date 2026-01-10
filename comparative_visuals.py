import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import Sim2RealBackbone
from model_scratch import Sim2RealBackbone as ScratchBackbone
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, det_curve
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import random
from tqdm import tqdm

LFW_DIR = "./dataset/lfw_aligned"
BASELINE_PATH = "./saved_models/baseline_model.pth"
PROPOSED_PATH = "./saved_models/best_model.pth"
SCRATCH_PATH = "./saved_models/best_scratch_model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "./final_comparison_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.style.use('ggplot')

def get_scores_and_embeddings(model_path, dataset, model_class=Sim2RealBackbone):
    print(f"Processing {os.path.basename(model_path)}...")
    model = model_class(pretrained=False).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    feats = []
    labels = []
    
    with torch.no_grad():
        for img, lbl in tqdm(loader):
            img = img.to(DEVICE)
            f = model(img)
            f = F.normalize(f)
            feats.append(f.cpu())
            labels.append(lbl)
            
    embeddings = torch.cat(feats)
    labels = torch.cat(labels)
    return embeddings, labels

def calculate_pair_scores(embeddings, labels, num_pairs=3000):
    pos_scores = []
    neg_scores = []
    labels_np = labels.numpy()
    
    unique_labels = np.unique(labels_np)
    idx_by_label = {lbl: np.where(labels_np == lbl)[0] for lbl in unique_labels}
    valid_classes = [l for l in unique_labels if len(idx_by_label[l]) > 1]
    
    print("Generating Pairs...")
    # Positives
    for _ in range(num_pairs):
        cls = random.choice(valid_classes)
        i1, i2 = random.sample(list(idx_by_label[cls]), 2)
        score = torch.dot(embeddings[i1], embeddings[i2]).item()
        pos_scores.append(score)
        
    # Negatives
    all_indices = np.arange(len(labels))
    for _ in range(num_pairs):
        i1 = random.choice(all_indices)
        i2 = random.choice(all_indices)
        if labels[i1] != labels[i2]:
            score = torch.dot(embeddings[i1], embeddings[i2]).item()
            neg_scores.append(score)
            
    return pos_scores, neg_scores

def plot_histogram_comparison(b_pos, b_neg, s_pos, s_neg, p_pos, p_neg):
    """ SCORE DISTRIBUTION """
    fig, axes = plt.subplots(1, 3, figsize=(21, 6), sharey=True)
    
    # Baseline
    sns.kdeplot(b_neg, fill=True, color='red', label='Different IDs', ax=axes[0], alpha=0.3)
    sns.kdeplot(b_pos, fill=True, color='green', label='Same IDs', ax=axes[0], alpha=0.3)
    axes[0].set_title(f"Baseline (Standard)")
    axes[0].set_xlabel("Cosine Similarity")
    axes[0].set_xlim(-0.5, 1.0)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Scratch
    sns.kdeplot(s_neg, fill=True, color='red', label='Different IDs', ax=axes[1], alpha=0.3)
    sns.kdeplot(s_pos, fill=True, color='green', label='Same IDs', ax=axes[1], alpha=0.3)
    axes[1].set_title(f"Scratch Model")
    axes[1].set_xlabel("Cosine Similarity")
    axes[1].set_xlim(-0.5, 1.0)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Proposed
    sns.kdeplot(p_neg, fill=True, color='red', label='Different IDs', ax=axes[2], alpha=0.3)
    sns.kdeplot(p_pos, fill=True, color='green', label='Same IDs', ax=axes[2], alpha=0.3)
    axes[2].set_title(f"Proposed (Texture-Regularized)")
    axes[2].set_xlabel("Cosine Similarity")
    axes[2].set_xlim(-0.5, 1.0)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle("Impact of Shape Bias on Score Separation", fontsize=16)
    plt.savefig(f"{OUTPUT_DIR}/cmp_1_histograms.png", dpi=300)
    print("Saved Histogram.")

def plot_det_comparison(b_pos, b_neg, s_pos, s_neg, p_pos, p_neg):
    """ DET CURVE (Security Standard) """
    # Baseline Metrics
    b_y_true = [1]*len(b_pos) + [0]*len(b_neg)
    b_y_score = b_pos + b_neg
    fpr_b, fnr_b, _ = det_curve(b_y_true, b_y_score)

    # Scratch Metrics
    s_y_true = [1]*len(s_pos) + [0]*len(s_neg)
    s_y_score = s_pos + s_neg
    fpr_s, fnr_s, _ = det_curve(s_y_true, s_y_score)
    
    # Proposed Metrics
    p_y_true = [1]*len(p_pos) + [0]*len(p_neg)
    p_y_score = p_pos + p_neg
    fpr_p, fnr_p, _ = det_curve(p_y_true, p_y_score)
    
    plt.figure(figsize=(8, 8))
    plt.plot(fpr_b, fnr_b, label="Baseline Model", color="blue", linestyle="--", lw=2)
    plt.plot(fpr_s, fnr_s, label="Scratch Model", color="green", linestyle="-.", lw=2)
    plt.plot(fpr_p, fnr_p, label="Proposed Model", color="darkorange", lw=2)
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('False Acceptance Rate (FAR)')
    plt.ylabel('False Rejection Rate (FRR)')
    plt.title('Detection Error Tradeoff (DET) Analysis')
    plt.grid(True, which="both", ls=":", alpha=0.4)
    plt.legend()
    plt.savefig(f"{OUTPUT_DIR}/cmp_2_det_curve.png", dpi=300)
    print("Saved DET Curve.")

def main():
    trans = transforms.Compose([
        transforms.Resize((112, 112)), 
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    dataset = datasets.ImageFolder(LFW_DIR, transform=trans)
    
    #  Data for Baseline
    emb_b, lbl_b = get_scores_and_embeddings(BASELINE_PATH, dataset)
    b_pos, b_neg = calculate_pair_scores(emb_b, lbl_b)

    #  Data for Scratch
    emb_s, lbl_s = get_scores_and_embeddings(SCRATCH_PATH, dataset, model_class=ScratchBackbone)
    s_pos, s_neg = calculate_pair_scores(emb_s, lbl_s)
    
    #  Data for Proposed
    emb_p, lbl_p = get_scores_and_embeddings(PROPOSED_PATH, dataset)
    p_pos, p_neg = calculate_pair_scores(emb_p, lbl_p)
    
    # Plot Comparison Charts
    plot_histogram_comparison(b_pos, b_neg, s_pos, s_neg, p_pos, p_neg)
    plot_det_comparison(b_pos, b_neg, s_pos, s_neg, p_pos, p_neg)
    
    print("\nCharts generated in folder:", OUTPUT_DIR)

if __name__ == "__main__":
    main()