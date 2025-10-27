# scripts/train_eval.py

import torch
import torch.nn as nn
from copy import deepcopy

# utils_config에서 필요한 요소 import
from .utils_config import matthews_corrcoef, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, device 


# -------------------- [A. 학습 루틴] --------------------
def train_model(model, optimizer, train_loader, val_loader, loss_fn, num_epochs=50, patience=10):
    best_val = float('inf'); best_state = None; bad = 0
    for epoch in range(num_epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                val_loss += loss_fn(model(x), y).item()
        val_loss /= len(val_loader)
        
        if val_loss < best_val:
            best_val = val_loss
            best_state = deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    return model

# -------------------- [B. 평가 루틴] --------------------
def eval_model(model, loader):
    model.eval()
    y_true, y_prob, y_pred = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy()
            pred = (probs > 0.5).astype(int)
            y_prob.extend(probs)
            y_pred.extend(pred)
            y_true.extend(y.numpy())
            
    cm = confusion_matrix(y_true, y_pred)
    
    # 클래스 불균형 등으로 인한 cm 크기 오류 처리
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        sensitivity = recall_score(y_true, y_pred, zero_division=0)
    else:
        specificity, sensitivity = 0.0, 0.0
        
    return {
        'accuracy': round(accuracy_score(y_true, y_pred), 3),
        'precision': round(precision_score(y_true, y_pred, zero_division=0), 3),
        'recall': round(sensitivity, 3),
        'f1': round(f1_score(y_true, y_pred, zero_division=0), 3),
        'roc_auc': round(roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else 0.0, 3),
        'mcc': round(matthews_corrcoef(y_true, y_pred), 3),
        'sensitivity': sensitivity,
        'specificity': specificity,
        'cm': cm
    }
