import os
import cv2
import torch
from facenet_pytorch import MTCNN
from torchvision import datasets
from tqdm import tqdm

RAW_LFW_DIR = "./data/lfw-deepfunneled" 
ALIGNED_LFW_DIR = "./dataset/lfw_aligned"  
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def align_lfw():
    if not os.path.exists(RAW_LFW_DIR):
        print(f"Error: {RAW_LFW_DIR} does not exist!")
        return

    print(f"Aligning LFW using MTCNN on {DEVICE}...")
    
    mtcnn = MTCNN(image_size=112, margin=0, min_face_size=20, device=DEVICE, post_process=False)
    
    dataset = datasets.ImageFolder(RAW_LFW_DIR)
    
    print(f"Found {len(dataset)} images to align.")
    
    for path, label_idx in tqdm(dataset.samples):
        
        person_name = os.path.basename(os.path.dirname(path))
        save_dir = os.path.join(ALIGNED_LFW_DIR, person_name)
        os.makedirs(save_dir, exist_ok=True)
        
        save_path = os.path.join(save_dir, os.path.basename(path))
        
        
        if os.path.exists(save_path): continue

        try:
            img = cv2.imread(path)
            if img is None: continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            mtcnn(img_rgb, save_path=save_path)
            
        except Exception:
            pass

    print("LFW Alignment Complete.")

if __name__ == "__main__":
    align_lfw()