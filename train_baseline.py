import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from model import Sim2RealBackbone, ArcFaceLoss
from tqdm import tqdm
import os
import numpy as np
from sklearn.model_selection import train_test_split

DATA_DIR = "./dataset/digiface_aligned"  
SAVE_DIR = "./saved_models"
BATCH_SIZE = 32      
EPOCHS = 20
LR = 0.01 

import matplotlib.pyplot as plt

def plot_training_history(history, filename="training_plot.png"):
    """
    Plots the Training vs Validation Accuracy/Loss curves.
    Essential for Section 8: Expected Outcomes.
    """
    epochs = range(1, len(history['loss']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(epochs, history['loss'], 'r-', marker='o', label="Training Loss")
    ax1.set_title("Training Loss Convergence")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("ArcFace Loss")
    ax1.set_xticks(np.arange(min(epochs), max(epochs)+1, 2))
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(epochs, history['train_acc'], 'b--', label="Training Acc (Blurred/Augmented)")
    ax2.plot(epochs, history['val_acc'], 'g-', marker='x', label="Validation Acc (Clean)")

    gap = max(history['val_acc']) - max(history['train_acc'])
    ax2.set_title(f"Sim2Real Generalization (Gap: +{gap:.1f}%)")
    
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy %")
    ax2.set_xticks(np.arange(min(epochs), max(epochs)+1, 2))
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    save_path = os.path.join(SAVE_DIR, filename)
    plt.savefig(save_path, dpi=300)
    print(f"Chart Saved to: {save_path}")
    


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- TRAINING STANDARD BASELINE MODEL on {device} ---")
    print("Configuration: No Gaussian Blur, No Color Jitter.")
    
    if not os.path.exists(DATA_DIR):
        print(f"ERROR: Data folder not found!")
        return
    baseline_transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.RandomHorizontalFlip(),
        
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    print("Loading Dataset...")
    full_ds_train = datasets.ImageFolder(root=DATA_DIR, transform=baseline_transform)
    full_ds_val   = datasets.ImageFolder(root=DATA_DIR, transform=val_transform)
    
    targets = full_ds_train.targets
    train_idx, val_idx = train_test_split(
        np.arange(len(targets)), 
        test_size=0.1, 
        shuffle=True, 
        stratify=targets
    )
    
    train_dataset = Subset(full_ds_train, train_idx)
    val_dataset   = Subset(full_ds_val, val_idx)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    
    num_classes = len(full_ds_train.classes)

    backbone = Sim2RealBackbone(pretrained=True).to(device)
    metric_crit = ArcFaceLoss(in_features=512, out_features=num_classes, s=30.0, m=0.50).to(device)

    optimizer = optim.SGD([
        {'params': backbone.parameters()},
        {'params': metric_crit.parameters()}
    ], lr=LR, momentum=0.9, weight_decay=5e-4) 
    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[8, 14], gamma=0.1)
    criterion = torch.nn.CrossEntropyLoss()
    
    os.makedirs(SAVE_DIR, exist_ok=True)
    best_acc = 0.0
    
    history = {'loss': [], 'train_acc': [], 'val_acc': []}
    for epoch in range(EPOCHS):
        backbone.train()
        metric_crit.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        loop = tqdm(train_loader, desc=f"Baseline Epoch {epoch+1}/{EPOCHS}")
        
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            features = backbone(images)
            outputs = metric_crit(features, labels)
            loss = criterion(outputs, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5.0)
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            loop.set_postfix(loss=loss.item(), acc=100 * correct / total)

        train_loss_epoch = total_loss / len(train_loader)
        train_acc_epoch = 100 * correct / total
   

        backbone.eval()
        metric_crit.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                features = backbone(images)
                outputs = metric_crit(features) 
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        print(f"Baseline Val Accuracy: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(backbone.state_dict(), os.path.join(SAVE_DIR, "baseline_model.pth"))
            print("--> Best Baseline Model Saved.")
        
        val_acc_epoch = 100 * val_correct / val_total
        history['loss'].append(train_loss_epoch)
        history['train_acc'].append(train_acc_epoch)
        history['val_acc'].append(val_acc_epoch)
        
        
        print(f"Stats: Loss={train_loss_epoch:.4f}, Train_Acc={train_acc_epoch:.2f}%, Val_Acc={val_acc_epoch:.2f}%")
        
        scheduler.step()

    plot_filename = "training_chart_baseline.png" 
    plot_training_history(history, plot_filename)

if __name__ == "__main__":
    main()