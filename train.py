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
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

def evaluate(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            
            # 1. Converte as saídas brutas em probabilidades (0 a 1)
            probs = F.softmax(outputs, dim=1)
            
            # 2. GUARDA OS DADOS (Certifique-se de que estas duas linhas estão bem indentadas aqui dentro)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            
    # 🚨 TRAVA DE SEGURANÇA: Se a lista continuar vazia, investigamos o Dataloader
    if len(all_probs) == 0:
        print("\n❌ [ERRO] O loop de validação terminou e 'all_probs' continua vazio!")
        print(f"Verifique se o seu 'val_loader' possui dados. Tamanho atual: {len(test_loader)} lotes.\n")
        return 0.0

    # Junta todos os lotes em matrizes estáveis do NumPy
    all_probs = np.vstack(all_probs)
    all_labels = np.concatenate(all_labels)
    
    # 3. Calcula o Macro ROC-AUC filtrando as classes do subset
    try:
        classes_presentes = np.unique(all_labels)
        probs_filtradas = all_probs[:, classes_presentes]
        
        roc_auc = roc_auc_score(
            all_labels, 
            probs_filtradas, 
            multi_class='ovr', 
            average='macro', 
            labels=classes_presentes
        )
    except Exception as e:
        print(f"\n[Erro no cálculo do ROC-AUC]: {e}")
        roc_auc = 0.0
        
    print(f'Test Loss: {test_loss/len(test_loader):.4f} | Competition ROC-AUC: {roc_auc:.4f}')
    return roc_auc