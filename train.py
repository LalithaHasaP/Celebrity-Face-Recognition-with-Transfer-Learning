import os
import random
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

def make_stratified_folds(root_dir, k=10):
    class_names = sorted([
        d for d in os.listdir(root_dir) 
        if os.path.isdir(os.path.join(root_dir, d))
    ])
    
    folds = [[] for _ in range(k)]
    
    print(f"Found {len(class_names)} classes. Distributing 10 images per class per fold...")

    for cls_idx, cls_name in enumerate(class_names):
        folder = os.path.join(root_dir, cls_name)
        
        #get all valid images for this actor
        img_files = glob.glob(os.path.join(folder, "*"))
        valid_exts = {'.jpg', '.jpeg', '.png'}
        img_files = [f for f in img_files if os.path.splitext(f)[1].lower() in valid_exts]

        #check to make sure 100 images exist
        if len(img_files) < 100:
            print(f"WARNING: {cls_name} has only {len(img_files)} images.")
            while len(img_files) < 100:
                img_files.extend(img_files[:100 - len(img_files)])
        
        #take the first 100 images
        img_files = img_files[:100]
        
        #distribute 10 images to each of the 10 folds
        for i in range(k):
            start = i * 10
            end = (i + 1) * 10
            batch = img_files[start:end]
            
            for img_path in batch:
                folds[i].append((img_path, cls_idx))
                
    return folds, class_names

class SimpleDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            img = Image.new('RGB', (224, 224)) 
            
        if self.transform:
            img = self.transform(img)
        return img, label


def get_model(num_classes):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    running_loss = 0.0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)

def test_model(model, loader, device):
    model.eval()
    all_true = []
    all_pred = []
    
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(1).cpu().numpy()
            
            all_pred.extend(preds)
            all_true.extend(labels.numpy())
            
    acc = accuracy_score(all_true, all_pred)
    return acc, all_true, all_pred

if __name__ == "__main__":
    DATA_DIR = "/kaggle/input/celebrity-faces-dataset/Celebrity Faces Dataset" 
    EPOCHS = 10          
    BATCH_SIZE = 32
    LR = 1e-4            
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")

    if not os.path.exists(DATA_DIR):
        print(f"ERROR: {DATA_DIR} not found.")
    else:
        folds, class_names = make_stratified_folds(DATA_DIR, k=10)
        num_classes = len(class_names)
        
        train_tf = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.6, 1.0)), 
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        test_tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        global_true = []
        global_pred = []
        fold_accs = []

        print(f"\nStarting 10-Fold Cross Validation on {num_classes} classes...")
        
        for i in range(10):
            print(f"\n--- Fold {i+1}/10 ---")
            
            test_samples = folds[i]
            train_samples = [item for f in range(10) if f != i for item in folds[f]]
            
            train_ds = SimpleDataset(train_samples, transform=train_tf)
            test_ds = SimpleDataset(test_samples, transform=test_tf)
            
            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
            test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
            
            model = get_model(num_classes).to(device)
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
                
            optimizer = torch.optim.Adam(model.parameters(), lr=LR)
            loss_fn = nn.CrossEntropyLoss()
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)

            for epoch in range(EPOCHS):
                loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
                scheduler.step()
                if epoch == EPOCHS - 1:
                    print(f"  Final Training Loss: {loss:.4f}")

            acc, f_true, f_pred = test_model(model, test_loader, device)
            fold_accs.append(acc)
            print(f"  Fold Accuracy: {acc:.4f}")

            df = pd.DataFrame({"fold": list(range(1, 11)), "accuracy": fold_accs})

            df.loc["mean"] = ["mean", df["accuracy"].mean()]
            df.to_csv("fold_metrics.csv", index=False)

            global_true.extend(f_true)
            global_pred.extend(f_pred)

        avg_acc = np.mean(fold_accs)
        print(f"FINAL RESULT (LOOCV Average): {avg_acc:.4f}")        





cm = confusion_matrix(global_true, global_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title(f'Confusion Matrix (Avg Acc: {avg_acc:.2f})')
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()
plt.close()
