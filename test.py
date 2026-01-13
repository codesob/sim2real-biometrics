import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import Sim2RealBackbone
from sklearn.manifold import TSNE
from sklearn.metrics import auc, roc_curve
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from tqdm import tqdm
from metrics_report import evaluate_biometrics 
import torchvision.transforms.functional as TF


LFW_DIR = "./dataset/lfw_aligned"
MODEL_PATH = "./saved_models/best_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
PAIRS_TO_GENERATE = 12000  

def load_model():
    print(f"Loading Model from {MODEL_PATH}...")
    model = Sim2RealBackbone(pretrained=False)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    
    model.eval()  
    return model

def generate_pairs(dataset):
    print("Generating Testing Pairs...")
    targets = np.array(dataset.targets)
    
    positive_pairs = []
    negative_pairs = []

    class_indices = {}
    for i, label in enumerate(targets):
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(i)

 
    valid_classes = [c for c in class_indices if len(class_indices[c]) > 1]

    
    while len(positive_pairs) < PAIRS_TO_GENERATE:
       
        cls = random.choice(valid_classes)
       
        idx1, idx2 = random.sample(class_indices[cls], 2)
        positive_pairs.append((idx1, idx2))

    
    all_indices = np.arange(len(dataset))
    while len(negative_pairs) < PAIRS_TO_GENERATE:
        
        idx1, idx2 = random.choice(all_indices), random.choice(all_indices)
   
        if targets[idx1] != targets[idx2]:
            negative_pairs.append((idx1, idx2))

    return positive_pairs, negative_pairs


def evaluate_roc(model, dataset, pos_pairs, neg_pairs):
    print(f"Running Metric Evaluation on {len(pos_pairs)*2} pairs...")
    
    similarities = []
    labels = [] 

    all_pairs = pos_pairs + neg_pairs
    all_labels = [1] * len(pos_pairs) + [0] * len(neg_pairs)
    
    unique_indices = list(set([p[0] for p in all_pairs] + [p[1] for p in all_pairs]))
    index_map = {idx: i for i, idx in enumerate(unique_indices)}

    subset = torch.utils.data.Subset(dataset, unique_indices)
    loader = DataLoader(subset, batch_size=64, shuffle=False)
    
    embeddings_map = []
    
    print("Extracting features (Batch Mode)...")
 
    with torch.no_grad():
        for imgs, _ in tqdm(loader):
            imgs = imgs.to(DEVICE)
            # 1. Forward Original
            feat_original = model(imgs)
            
            # 2. Forward Flipped
            imgs_flipped = TF.hflip(imgs)
            feat_flipped = model(imgs_flipped)
            
            feats = feat_original + feat_flipped
            feats = F.normalize(feats, p=2, dim=1)
            
            embeddings_map.append(feats.cpu())
            
    embeddings_map = torch.cat(embeddings_map, dim=0)

    print("Calculating distances...")
    for idx1, idx2 in all_pairs:
        
        i1 = index_map[idx1]
        i2 = index_map[idx2]
        
        vec1 = embeddings_map[i1]
        vec2 = embeddings_map[i2]
      
        sim = torch.dot(vec1, vec2).item()
        
        similarities.append(sim)

    labels = np.array(all_labels)
    similarities = np.array(similarities)
    
    evaluate_biometrics(np.array(all_labels), np.array(similarities), far_target=0.001)
    
    if np.isnan(similarities).any():
        print("WARNING: NaNs detected in similarities! Replacing with 0.")
        similarities = np.nan_to_num(similarities)

    fpr, tpr, thresholds = roc_curve(labels, similarities)
    roc_auc = auc(fpr, tpr)
    
    max_acc = 0
    best_thresh = 0
    for thresh in thresholds:
        pred_labels = (similarities >= thresh)
        acc = np.mean((pred_labels == labels))
        if acc > max_acc:
            max_acc = acc
            best_thresh = thresh

    return fpr, tpr, roc_auc, max_acc, best_thresh

def plot_tsne(model, dataset, filename="./plot_image/lfw_tsne_plot.png"):
    print("\nGenerating t-SNE Clustering Plot...")
    
    targets = np.array(dataset.targets)
    unique, counts = np.unique(targets, return_counts=True)
    top_10_indices = np.argsort(counts)[-10:] 
    target_ids = unique[top_10_indices]
    
    subset_indices = []
    plot_labels = []
    
    for i, real_label_id in enumerate(target_ids):
        indices = np.where(targets == real_label_id)[0]
        subset_indices.extend(indices)
        plot_labels.extend([i] * len(indices))

    subset_dataset = torch.utils.data.Subset(dataset, subset_indices)
    loader = DataLoader(subset_dataset, batch_size=32, shuffle=False)

    features = []
    
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(DEVICE)
            emb = model(imgs)
            emb = F.normalize(emb)
            features.append(emb.cpu().numpy())
    
    features = np.concatenate(features, axis=0)
    
    tsne = TSNE(n_components=2, perplexity=30, init='pca', random_state=42)
    embedded = tsne.fit_transform(features)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embedded[:,0], embedded[:,1], c=plot_labels, cmap='tab10', s=40)
    plt.legend(handles=scatter.legend_elements()[0], labels=[f"ID {x}" for x in range(10)], title="Identities")
    plt.title("t-SNE Clustering of Real LFW Faces\n(Trained on Synthetic Only)")
    plt.grid(True, alpha=0.3)
    plt.savefig(filename)
    print(f"Saved t-SNE plot to {filename}")

def main():

    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    if not os.path.exists(LFW_DIR):
        print("LFW Directory not found! Please check path.")
        return

    full_lfw = datasets.ImageFolder(LFW_DIR, transform=transform)
    
    model = load_model()

    pos, neg = generate_pairs(full_lfw)
    fpr, tpr, roc_auc, max_acc, thresh = evaluate_roc(model, full_lfw, pos, neg)
    
    print("\n" + "="*40)
    print(f"FINAL LFW BENCHMARK RESULTS")
    print("="*40)
    print(f"Model: {MODEL_PATH}")
    print(f"Validation Pairs: {len(pos) + len(neg)}")
    print(f"Area Under Curve (AUC): {roc_auc:.4f}")
    print(f"Best Accuracy:          {max_acc*100:.2f}%")
    print(f"Optimal Threshold:      {thresh:.4f}")
    print("="*40)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (Sim2Real)')
    plt.legend(loc="lower right")
    plt.savefig('./plot_image/lfw_roc_curve.png')
    print("Saved ROC Curve to lfw_roc_curve.png")

    plot_tsne(model, full_lfw)

if __name__ == "__main__":
    main()