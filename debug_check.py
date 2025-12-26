import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import Sim2RealBackbone
import matplotlib.pyplot as plt
import numpy as np
import os


LFW_DIR = "./dataset/lfw_aligned"
MODEL_PATH = "./saved_models/best_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def check_system():
    print(f"DEBUG: Checking path: {os.path.abspath(LFW_DIR)}")
    
    if not os.path.exists(LFW_DIR):
        print("CRITICAL ERROR: LFW folder not found.")
        return

    simple_transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor()
    ])
    
    try:
        dataset = datasets.ImageFolder(LFW_DIR, transform=simple_transform)
        print(f"DEBUG: Found {len(dataset)} images in dataset.")
    except Exception as e:
        print(f"CRITICAL ERROR loading dataset: {e}")
        return
        
    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    images_vis, labels_vis = next(iter(loader))
    
    print("\n--- VISUAL CHECK ---")
    print("If you don't see two faces pop up, your data path is wrong.")
    plt.figure(figsize=(8,4))
    
    
    plt.subplot(1, 2, 1)
  
    plt.imshow(images_vis[0].permute(1, 2, 0))
    plt.title(f"Input 1 (Label {labels_vis[0].item()})")
   
    plt.subplot(1, 2, 2)
    plt.imshow(images_vis[1].permute(1, 2, 0))
    plt.title(f"Input 2 (Label {labels_vis[1].item()})")
    plt.show()

    print("\n--- MODEL INTELLIGENCE CHECK ---")
    
    norm_transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) 
    ])
    dataset = datasets.ImageFolder(LFW_DIR, transform=norm_transform)
    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    try:
        model = Sim2RealBackbone(pretrained=False).to(DEVICE)
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.eval() 
        print("DEBUG: Weights loaded successfully.")
    except Exception as e:
        print(f"CRITICAL ERROR loading model: {e}")
        return

    images, _ = next(iter(loader))
    images = images.to(DEVICE)
    
    with torch.no_grad():
        embeddings = model(images)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

    vec1 = embeddings[0].cpu().numpy()
    vec2 = embeddings[1].cpu().numpy()

    print("\nVECTOR INSPECTION:")
    print(f"Vector 1 First 5 nums: {vec1[:5]}")
    print(f"Vector 2 First 5 nums: {vec2[:5]}")
    
    if np.allclose(vec1, vec2, atol=1e-4):
        print("FAIL: The model is outputting identical vectors for everyone. Training Failed.")
    else:
        print("PASS: The model is outputting unique vectors.")
        

    if np.count_nonzero(vec1) == 0:
        print("FAIL: The model is outputting ZEROs.")
        
    similarity = np.dot(vec1, vec2)
    print(f"\nCosine Similarity between random pair: {similarity:.4f}")
    print("(Should be < 0.8 for different people)")

if __name__ == "__main__":
    check_system()