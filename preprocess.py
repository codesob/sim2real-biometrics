import os
import cv2
import torch
from facenet_pytorch import MTCNN
from tqdm import tqdm
from torchvision import datasets


RAW_DATA_FOLDERS = [
    "./data/subjects_0-1999_72_imgs",
    "./data/subjects_2000-3999_72_imgs",
    "./data/subjects_4000-5999_72_imgs"
]


PROCESSED_DIR = "./dataset/digiface_aligned" 

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def align_dataset():
    print(f"Running MTCNN Face Alignment on {DEVICE}...")
    
    
    mtcnn = MTCNN(image_size=112, margin=0, min_face_size=20, device=DEVICE, post_process=False)
    
  
    for raw_folder in RAW_DATA_FOLDERS:
        print(f"\n---> Processing Source: {raw_folder}")
        
        if not os.path.exists(raw_folder):
            print(f"Skipping {raw_folder} (Not found)")
            continue

        
        dataset = datasets.ImageFolder(raw_folder)
        print(f"Found {len(dataset)} images in this chunk.")

        count = 0
        err_count = 0
        
        for path, label_idx in tqdm(dataset.samples):
           
            class_name = os.path.basename(os.path.dirname(path))
            
            save_dir = os.path.join(PROCESSED_DIR, class_name)
            os.makedirs(save_dir, exist_ok=True)
            
            save_path = os.path.join(save_dir, os.path.basename(path))
            
            if os.path.exists(save_path):
                continue
                
            try:
                img = cv2.imread(path)
                if img is None: continue
             
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                img_cropped = mtcnn(img_rgb, save_path=save_path)
                
                if img_cropped is not None:
                    count += 1
                else:
                  
                    resized = cv2.resize(img, (112, 112))
                    cv2.imwrite(save_path, resized)
                    err_count += 1
                    
            except Exception as e:
                pass
        
        print(f"Chunk Complete. Saved to {PROCESSED_DIR}")

if __name__ == "__main__":
    align_dataset()