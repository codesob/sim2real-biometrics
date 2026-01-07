import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import Sim2RealBackbone, ArcFaceLoss
from tqdm import tqdm
import os

DATA_DIR = "./dataset/digiface_aligned"  
SAVE_DIR = "./saved_models"
BATCH_SIZE = 32      
EPOCHS = 10
LR = 0.01

def main():
   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Started Training using: {device}")
    
    if not os.path.exists(DATA_DIR):
        print(f"ERROR: Data folder not found at {DATA_DIR}. Please unzip DigiFace there.")
        return

   
    train_transforms = transforms.Compose([
        transforms.Resize((112, 112)),
        
        
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2.0))
        ], p=0.5),
        
    
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
      

        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

   
    print("Loading Dataset...")
    train_dataset = datasets.ImageFolder(root=DATA_DIR, transform=train_transforms)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    
    num_classes = len(train_dataset.classes)
    print(f"Found {len(train_dataset)} images belonging to {num_classes} identities.")

   
    backbone = Sim2RealBackbone(pretrained=True).to(device)
    metric_crit = ArcFaceLoss(in_features=512, out_features=num_classes).to(device)

    
    optimizer = optim.SGD([
        {'params': backbone.parameters()},
        {'params': metric_crit.parameters()}
    ], lr=LR, momentum=0.9, weight_decay=5e-4)
    
    criterion = torch.nn.CrossEntropyLoss()

    
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    for epoch in range(EPOCHS):
        backbone.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            
           
            features = backbone(images)
            outputs = metric_crit(features, labels)
            loss = criterion(outputs, labels)
            
         
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
         
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            loop.set_postfix(loss=loss.item(), acc=100 * correct / total)

        print(f"Epoch {epoch+1} Complete. Avg Loss: {total_loss / len(train_loader):.4f}")

    
    save_path = os.path.join(SAVE_DIR, "sim2real_model.pth")
    torch.save(backbone.state_dict(), save_path)
    print(f"Model saved successfully to {save_path}")

if __name__ == "__main__":
    main()