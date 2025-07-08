# -*- coding: utf-8 -*-

import torch
from torch import nn
from sklearn.model_selection import KFold
from torch.utils.data import TensorDataset, DataLoader,Subset
import numpy as np
from sklearn.model_selection import KFold
from model import ACRNN



def train_model(model, train_loader, device, epochs=500):
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    best_loss = float('inf')
    best_model_arousal = None

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            output = model(xb)
            loss = criterion(output, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
            
        avg_epoch_loss = epoch_loss / len(train_loader.dataset)
        if (epoch+1)%100==0:
            print(f'Epoch {epoch+1} average loss:{avg_epoch_loss}')        
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            best_model_arousal = model.state_dict()

    return best_model_arousal


def cross_validate_model():
    device = torch.device('cuda')
    all_subject_acc = []
    
    for subject_id in range(1,33):
        print(f"\n=== Subject {subject_id} ===")
        data = torch.load(f'../../data/data_preprocessed_ACRNN/s{subject_id:02d}.pth')
        X_subset=data['x']
        y_subset=data['y'][:,1]
        X_subset = X_subset.squeeze(1)


        kf = KFold(n_splits=10, shuffle=True, random_state=42) 
        all_acc = []
            
        for fold, (train_val_idx, test_idx) in enumerate(kf.split(X_subset)):
            print(f"\n========== Fold {fold + 1}/10 ==========")
        
            X_train_val, y_train_val = X_subset[train_val_idx], y_subset[train_val_idx]
            X_test, y_test = X_subset[test_idx], y_subset[test_idx]
            
            train_ds = TensorDataset(X_train_val, y_train_val)
            test_ds = TensorDataset(X_test, y_test)
        
            train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
            test_loader = DataLoader(test_ds, batch_size=16)
        

            model = ACRNN(reduce = 2, k = 40).to(device)
            best_model = train_model(model, train_loader, device)
        
    
            model.load_state_dict(best_model)
            model.eval()
            all_preds, all_targets = [], []
        
            with torch.no_grad():
                for xb, yb in test_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    outputs = model(xb)
                    preds = torch.argmax(outputs, dim=1)
                    all_preds.append(preds)
                    all_targets.append(yb)
            
            all_preds = torch.cat(all_preds)
            all_targets = torch.cat(all_targets)
            acc = (all_preds == all_targets).sum().item()/len(all_preds)
            
            print(f"Fold {fold + 1} - Accuracy: {acc:.4f}")
            all_acc.append(acc)
            print("=======================================\n")
        
        mean_acc = np.mean(all_acc)
        std_acc = np.std(all_acc)
        all_subject_acc.append(mean_acc)
        print(f"\n Final 10-Fold Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
        
    overall_mean = np.mean(all_subject_acc)
    overall_std = np.std(all_subject_acc)
    print(f"\n=== Overall Subject-Dependent Accuracy: {overall_mean:.4f} ± {overall_std:.4f} ===")
    
cross_validate_model()