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
import torch.nn.functional as F
import sklearn.metrics

#data

def train_model(model, train_loader, val_loader, criterion, optimizer, device,epochs=10,step_size=2, gamma=0.5):
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    checkpoint_dir = '/kaggle/working/checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
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
        current_roc_auc= evaluate(model,val_loader,criterion,device)
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_roc_auc': max(current_roc_auc, best_roc_auc)
        }
        
        torch.save(checkpoint, os.path.join(checkpoint_dir, 'last_checkpoint.pth'))
     
        if current_roc_auc > best_roc_auc:
            print(f"Novo Recorde Detectado! ROC-AUC subiu de {best_roc_auc:.4f} para {current_roc_auc:.4f}.")
            print("Salvando o melhor modelo em: 'checkpoints/best_model.pth'")
            best_roc_auc = current_roc_auc
            torch.save(checkpoint, os.path.join(checkpoint_dir, 'best_model.pth'))
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
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            
            # Transforma os logits brutos em probabilidades (0 a 1)
            probs = F.softmax(outputs, dim=1)
            
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            
    if len(all_probs) == 0:
        print("\n [ERRO] O val_loader está vazio.")
        return 0.0

    # Consolida os arrays de todos os lotes
    all_probs = np.vstack(all_probs)
    all_labels = np.concatenate(all_labels)

    num_classes = all_probs.shape[1]
    
    # 1. Transforma as labels 1D em uma matriz One-Hot (Equivalente ao DataFrame 'solution')
    # Formato final: [N_amostras, Num_classes] contendo apenas 0s e 1s
    solution_matrix = np.eye(num_classes)[all_labels]
    submission_matrix = all_probs  # Suas probabilidades calculadas
    
    # 2. Identifica quais colunas possuem pelo menos um True Positive (solution.sum(axis=0) > 0)
    solution_sums = solution_matrix.sum(axis=0)
    scored_columns = np.where(solution_sums > 0)[0]
    
    # 3. Filtra as matrizes deixando apenas as colunas válidas (Exatamente como o Kaggle faz)
    y_true = solution_matrix[:, scored_columns]
    y_score = submission_matrix[:, scored_columns]
    
    # 4. Calcula o Macro ROC-AUC final
    try:
        # Passar matrizes 2D para o sklearn força ele a tratar como multi-label,
        # eliminando aquela checagem chata de as linhas precisarem somar 1.0!
        roc_auc = sklearn.metrics.roc_auc_score(y_true, y_score, average='macro')
    except Exception as e:
        print(f"\n[Erro inesperado na métrica]: {e}")
        roc_auc = 0.0
        
    print(f'Test Loss: {test_loss/len(test_loader):.4f} | Competition ROC-AUC: {roc_auc:.4f}')
    return roc_auc