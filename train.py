from dataset_bird import BirdDataset
from sklearn.metrics import roc_auc_score
import utils
import torch.nn as nn
import torch
import numpy as np
import pandas as pd
import os
import librosa 
# data augumentations
from utils import DataPipelines, CFG, data_transforms
from sklearn.preprocessing import LabelEncoder
#data

def train_model(model, train_loader, val_loader, criterion, optimizer, device,epochs=10,step_size=2, gamma=0.5):
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}')
        scheduler.step()
        evaluate(model,val_loader,criterion,device)
    return model
def evaluate_oldest(model, test_loader,  criterion, device):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            
            # Atualiza os contadores (labels já é o índice correto!)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Test Loss: {test_loss/len(test_loader):.4f}, Accuracy: {100 * correct / total:.2f}%')
def evaluate(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            
            # 1. Transforma os logits brutos em probabilidades (0 a 1) via Softmax
            probs = torch.nn.functional.softmax(outputs, dim=1)
            
            # 2. Guarda tudo em formato NumPy para o scikit-learn calcular depois
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            
    # Junta todos os lotes (batches) em matrizes gigantes
    all_probs = np.vstack(all_probs)
    all_labels = np.concatenate(all_labels)
    
    # 3. Calcula o Macro ROC-AUC (estratégia One-vs-Rest para classes múltiplas)
    # Nota: Usamos multi_class='ovr' porque avalia cada classe contra as outras.
    try:
        # Nota: O Kaggle ignora classes que não aparecem no set de teste. 
        # Passar as labels existentes evita erros se alguma classe sumir no split.
        classes_presentes = np.unique(all_labels)
        roc_auc = roc_auc_score(
            all_labels, 
            all_probs, 
            multi_class='ovr', 
            average='macro', 
            labels=classes_presentes
        )
    except Exception as e:
        roc_auc = 0.0 # Caso ocorra algum problema de amostragem nos primeiros lotes
        
    print(f'Test Loss: {test_loss/len(test_loader):.4f} | Competition ROC-AUC: {roc_auc:.4f}')
    return roc_auc