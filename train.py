import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from model import Sim2RealBackbone, ArcFaceLoss
from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# --- CONFIGURATION ---
DATA_DIR = "./dataset/digiface_aligned"  
SAVE_DIR = "./plot_image"
BATCH_SIZE = 32      
EPOCHS = 20           
LR = 0.01             

def plot_training_history(history):
    print("Generating Professional Training Dynamics Chart...")
    epochs = range(1, len(history['loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(epochs, history['loss'], 'r-o', markersize=4, label="ArcFace Loss")
    ax1.set_title("Proposed Model: Training Loss")
    ax1.set_xlabel("Epochs")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(epochs, history['train_acc'], 'b--', alpha=0.7, label="Train (Robust)")
    ax2.plot(epochs, history['val_acc'], 'g-x', linewidth=2, label="Val (Clean)")
    ax2.set_title("Proposed Model: Sim2Real Gap")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy (%)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.savefig("proposed_strong_results.png", dpi=300)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- TRAINING PROPOSED STRONG MODEL (20 EPOCHS) on {device} ---")
    
    train_transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20), 
        transforms.RandomPerspective(distortion_scale=0.25, p=0.5),
        transforms.RandomApply([
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
            transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0))
        ], p=0.7),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.1), ratio=(0.3, 3.3))
    ])

    val_transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # Data Splitting
    full_ds_train = datasets.ImageFolder(root=DATA_DIR, transform=train_transform)
    full_ds_val   = datasets.ImageFolder(root=DATA_DIR, transform=val_transform)
    targets = full_ds_train.targets
    train_idx, val_idx = train_test_split(np.arange(len(targets)), test_size=0.1, shuffle=True, stratify=targets)
    train_loader = DataLoader(Subset(full_ds_train, train_idx), batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(Subset(full_ds_val, val_idx), batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    
    num_classes = len(full_ds_train.classes)
    
    backbone = Sim2RealBackbone(pretrained=True).to(device)
    metric_crit = ArcFaceLoss(in_features=512, out_features=num_classes, s=64.0, m=0.50).to(device)

    optimizer = optim.SGD([
        {'params': backbone.parameters(), 'lr': LR * 0.1}, 
        {'params': metric_crit.parameters(), 'lr': LR}
    ], momentum=0.9, weight_decay=1e-3) 
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    
    os.makedirs(SAVE_DIR, exist_ok=True)
    best_acc = 0.0 
    history = {'loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(EPOCHS):
        backbone.train()
        t_loss, t_correct, t_total = 0.0, 0, 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/20")
        
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            features = backbone(images)
            outputs = metric_crit(features, labels)
            loss = criterion(outputs, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5.0)
            optimizer.step()
            
            t_loss += loss.item()
            _, pred = torch.max(outputs.data, 1)
            t_total += labels.size(0)
            t_correct += (pred == labels).sum().item()
            loop.set_postfix(loss=loss.item(), acc=100 * t_correct / t_total)

        # Validation
        backbone.eval()
        v_correct, v_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                features = backbone(images)
                outputs = metric_crit(features, labels) 
                _, pred = torch.max(outputs.data, 1)
                v_total += labels.size(0)
                v_correct += (pred == labels).sum().item()
        
        val_acc = 100 * v_correct / v_total
        history['loss'].append(t_loss / len(train_loader))
        history['train_acc'].append(100 * t_correct / t_total)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1} -> Train_Acc: {history['train_acc'][-1]:.2f}%, Val_Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(backbone.state_dict(), os.path.join(SAVE_DIR, "proposed_strong_model.pth"))
            print("--> Saving New Best Proposed Model.")

        scheduler.step()

    plot_training_history(history)

if __name__ == "__main__":
    main()