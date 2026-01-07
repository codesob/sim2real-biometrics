import torch
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import numpy as np

# CONFIG
DATA_DIR = "./dataset/digiface_aligned"

def show_augmentation():
    # 1. Define your specific "Novelty" transform
    blur_trans = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=2.0), # MAX BLUR
        transforms.ColorJitter(0.2, 0.2, 0.2)
    ])
    
    clean_trans = transforms.Resize((112, 112))

    # 2. Grab an image
    ds = datasets.ImageFolder(DATA_DIR)
    # Pick a random image
    img, _ = ds[500] 

    # 3. Apply Transforms
    clean_img = clean_trans(img)
    blurred_img = blur_trans(img)

    # 4. Plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(clean_img)
    axes[0].set_title("Input 1: Raw Synthetic Texture\n(Perfect, Digital Skin)")
    axes[0].axis('off')

    axes[1].imshow(blurred_img)
    axes[1].set_title("Input 2: Frequency Regularized\n(Forced Shape Bias)")
    axes[1].axis('off')

    plt.savefig("novelty_visual_proof.png", dpi=300)
    print("Saved 'novelty_visual_proof.png'")

if __name__ == "__main__":
    show_augmentation()