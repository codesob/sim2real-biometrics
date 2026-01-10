import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import Sim2RealBackbone
from model_scratch import Sim2RealBackbone as ScratchBackbone
from sklearn.metrics import auc, roc_curve
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from tqdm import tqdm

LFW_DIR = "./dataset/lfw_aligned"
BASELINE_PATH = "./saved_models/baseline_model.pth"
PROPOSED_PATH = "./saved_models/best_model.pth" 
SCRATCH_PATH = "./saved_models/best_scratch_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PAIRS_TO_TEST = 12000

def get_roc_data(model_path, dataset, pos_pairs, neg_pairs, label_name, model_class):
    print(f"\n--- Testing {label_name} ---")
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found.")
        return None, None, 0.0

    model = model_class(pretrained=False).to(DEVICE)
    try:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    except RuntimeError as e:
        print(f"Error loading state dict for {label_name}: {e}")
        return None, None, 0.0

    model.eval()
  
    all_pairs = pos_pairs + neg_pairs
    all_labels = [1] * len(pos_pairs) + [0] * len(neg_pairs)
    
    unique_indices = list(set([p[0] for p in all_pairs] + [p[1] for p in all_pairs]))
    index_map = {idx: i for i, idx in enumerate(unique_indices)}
    subset = torch.utils.data.Subset(dataset, unique_indices)
    loader = DataLoader(subset, batch_size=64, shuffle=False)
    
    feats_list = []
    with torch.no_grad():
        for imgs, _ in tqdm(loader, disable=True):
            feats = model(imgs.to(DEVICE))
            feats_list.append(F.normalize(feats).cpu())
    
    embedding_bank = torch.cat(feats_list)
    
    scores = []
    for idx1, idx2 in all_pairs:
        v1 = embedding_bank[index_map[idx1]]
        v2 = embedding_bank[index_map[idx2]]
        scores.append(torch.dot(v1, v2).item())
    
    fpr, tpr, _ = roc_curve(all_labels, scores)
    roc_auc = auc(fpr, tpr)
    print(f"{label_name} AUC: {roc_auc:.4f}")
    
    return fpr, tpr, roc_auc

def main():
    if not os.path.exists(PROPOSED_PATH):
        print(f"Proposed path {PROPOSED_PATH} not found.")
        return

    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    full_lfw = datasets.ImageFolder(LFW_DIR, transform=transform)
    
    print("Generating Testing Pairs...")
    targets = np.array(full_lfw.targets)
  
    pos_pairs, neg_pairs = [], []
    class_indices = {}
    for i, label in enumerate(targets):
        class_indices.setdefault(label, []).append(i)
    valid_classes = [c for c in class_indices if len(class_indices[c]) > 1]
    
    for _ in range(PAIRS_TO_TEST):
        cls = random.choice(valid_classes)
        pos_pairs.append(tuple(random.sample(class_indices[cls], 2)))
        
        idx1, idx2 = random.sample(range(len(full_lfw)), 2)
        while targets[idx1] == targets[idx2]: idx1, idx2 = random.sample(range(len(full_lfw)), 2)
        neg_pairs.append((idx1, idx2))

    fpr_b, tpr_b, auc_b = get_roc_data(BASELINE_PATH, full_lfw, pos_pairs, neg_pairs, "Baseline", Sim2RealBackbone)
    fpr_p, tpr_p, auc_p = get_roc_data(PROPOSED_PATH, full_lfw, pos_pairs, neg_pairs, "Proposed", Sim2RealBackbone)
    fpr_s, tpr_s, auc_s = get_roc_data(SCRATCH_PATH, full_lfw, pos_pairs, neg_pairs, "Scratch", ScratchBackbone)

    plt.figure(figsize=(10, 8))
    if fpr_b is not None:
        plt.plot(fpr_b, tpr_b, color='navy', lw=3, linestyle='--', label=f'Baseline (No Blur) AUC = {auc_b:.2f}')
    if fpr_p is not None:
        plt.plot(fpr_p, tpr_p, color='crimson', lw=3, label=f'Proposed (With Blur) AUC = {auc_p:.2f}')
    if fpr_s is not None:
        plt.plot(fpr_s, tpr_s, color='darkorange', lw=3, linestyle='-.', label=f'Scratch AUC = {auc_s:.2f}')
        
    plt.plot([0, 1], [0, 1], color='gray', linestyle=':')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Sim2Real Gap Analysis: Comparison with Scratch')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig('./final_comparison_plots/final_comparison_chart_with_scratch.png')
    print("Graph saved to final_comparison_chart_with_scratch.png")

if __name__ == "__main__":
    main()