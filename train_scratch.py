import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from model_scratch import Sim2RealBackbone, ArcFaceLoss
from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

DATA_DIR = "./dataset/digiface_aligned"  
SAVE_DIR = "./saved_models"
BATCH_SIZE = 32      
EPOCHS = 20
LR = 0.1 

def plot_training_history(history):
    print("Generating Training Charts from Actual Data...")
    
    epochs = range(1, len(history['loss']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(epochs, history['loss'], 'r-', marker='o', label="ArcFace Loss")
    ax1.set_title("Real Training Loss Convergence")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.set_xticks(np.arange(min(epochs), max(epochs)+1, 2)) # Steps of 2 to avoid overlap
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(epochs, history['train_acc'], 'b--', label="Train (Blurred/Augmented)")
    ax2.plot(epochs, history['val_acc'], 'g-', marker='x', label="Val (Clean)")
    ax2.set_title("Sim2Real Generalization Gap")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_xticks(np.arange(min(epochs), max(epochs)+1, 2)) # Steps of 2 to avoid overlap
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    save_path = os.path.join(SAVE_DIR, "training_dynamics_chart.png")
    plt.savefig(save_path, dpi=300)
    print(f"Chart Saved to: {save_path}")
    


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Started Training using: {device}")
    
    if not os.path.exists(DATA_DIR):
        print(f"ERROR: Data folder not found!")
        return

    train_transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2.0))
        ], p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.3, 3.3))
    ])

    val_transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    full_ds_train = datasets.ImageFolder(root=DATA_DIR, transform=train_transform)
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
        {'params': backbone.parameters(), 'lr': LR * 0.1},
        {'params': metric_crit.parameters(), 'lr': LR}
    ], momentum=0.9, weight_decay=5e-4) 
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    criterion = torch.nn.CrossEntropyLoss()
    os.makedirs(SAVE_DIR, exist_ok=True)
    best_acc = 0.0
    patience = 5
    no_improve_epochs = 0

    history = {'loss': [], 'train_acc': [], 'val_acc': []}


    for epoch in range(EPOCHS):
        backbone.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
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
            
            # Calculate accuracy using raw similarities (without margin)
            with torch.no_grad():
                raw_outputs = metric_crit(features)
                _, predicted = torch.max(raw_outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            loop.set_postfix(loss=loss.item(), acc=100 * correct / total)

        epoch_loss = total_loss / len(train_loader)
        epoch_train_acc = 100 * correct / total
        history['loss'].append(epoch_loss)
        history['train_acc'].append(epoch_train_acc)

        backbone.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                features = backbone(images)
                outputs = metric_crit(features) # No label -> Raw similarities
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        
        history['val_acc'].append(val_acc)
        
        print(f"Stats: Loss={epoch_loss:.4f}, Train_Acc={epoch_train_acc:.2f}%, Val_Acc={val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(backbone.state_dict(), os.path.join(SAVE_DIR, "best_model.pth"))
            print("--> Best Model Saved.")
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        scheduler.step()

    torch.save(backbone.state_dict(), os.path.join(SAVE_DIR, "final_model.pth"))
    
    plot_training_history(history)

if __name__ == "__main__":
    main()