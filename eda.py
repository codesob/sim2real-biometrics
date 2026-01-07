import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from tqdm import tqdm
import random

SYN_DIR = "./dataset/digiface_aligned"
REAL_DIR = "./dataset/lfw_aligned"
OUTPUT_DIR = "./eda_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)
plt.style.use('ggplot')

def get_image_paths(directory, limit=5000):
    all_paths = []
    print(f"Scanning {directory}...")
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(('.jpg', '.png', '.jpeg')):
                all_paths.append(os.path.join(root, file))
    
    if len(all_paths) > limit:
        return random.sample(all_paths, limit)
    return all_paths

def plot_visual_comparison(syn_paths, real_paths):
    print("Generating Visual Comparison Grid...")
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    
    for i in range(5):
        img = cv2.imread(syn_paths[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[0, i].imshow(img)
        axes[0, i].axis('off')
        if i == 2: axes[0, i].set_title("Source Domain: Synthetic (DigiFace)", fontsize=14, pad=10)

    for i in range(5):
        img = cv2.imread(real_paths[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[1, i].imshow(img)
        axes[1, i].axis('off')
        if i == 2: axes[1, i].set_title("Target Domain: Real-World (LFW)", fontsize=14, pad=10)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/eda_1_visual_gap.png", dpi=300)
    print("Saved Chart 1 (Visual Gap).")

def plot_pixel_distribution(syn_paths, real_paths):
    print("Generating Pixel Intensity Histogram...")
    
    syn_pixels = []
    real_pixels = []
    
    print("Processing Synthetic Pixels...")
    for p in tqdm(syn_paths):
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is not None: syn_pixels.append(img.flatten())
            
    print("Processing Real Pixels...")
    for p in tqdm(real_paths):
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is not None: real_pixels.append(img.flatten())

    syn_pixels = np.concatenate(syn_pixels)
    real_pixels = np.concatenate(real_pixels)
    
    plt.figure(figsize=(10, 6))
    sns.kdeplot(syn_pixels, fill=True, label='Synthetic Data', color='#FF6B6B')
    sns.kdeplot(real_pixels, fill=True, label='Real Data (LFW)', color='#4ECDC4')
    
    plt.title("Domain Gap Quantification: Pixel Intensity Distribution")
    plt.xlabel("Pixel Value (0-255)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{OUTPUT_DIR}/eda_2_pixel_histogram.png", dpi=300)
    print("Saved Chart 2 (Pixel Hist).")

def plot_class_imbalance():
    print("Generating Class Imbalance Chart...")
    
    
    counts = []
    for dirname in os.listdir(REAL_DIR):
        dirpath = os.path.join(REAL_DIR, dirname)
        if os.path.isdir(dirpath):
            n_imgs = len(os.listdir(dirpath))
            counts.append(n_imgs)
            
    counts = np.array(counts)
    

    plt.figure(figsize=(12, 5))
    
    plt.hist(counts, bins=30, color='purple', alpha=0.7, log=True)
    
    plt.title("LFW Dataset: The 'Long Tail' Problem")
    plt.xlabel("Images per Identity")
    plt.ylabel("Number of Identities (Log Scale)")
    plt.grid(True, which="both", ls="--", alpha=0.2)
    plt.text(50, 100, "Most identities have\nonly 1 image", fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.savefig(f"{OUTPUT_DIR}/eda_3_class_imbalance.png", dpi=300)
    print("Saved Chart 3 (Class Balance).")

def main():
    print("--- Starting Exploratory Data Analysis ---")
    
    syn_files = get_image_paths(SYN_DIR, limit=2000)
    real_files = get_image_paths(REAL_DIR, limit=2000)
    
    if len(syn_files) == 0 or len(real_files) == 0:
        print("Error: Images not found. Check path config.")
        return

    plot_visual_comparison(syn_files[:5], real_files[:5]) # Pass exactly 5
    plot_pixel_distribution(syn_files, real_files)
    plot_class_imbalance()
    
    print("\nEDA Complete. Check the 'eda_results' folder.")

if __name__ == "__main__":
    main()