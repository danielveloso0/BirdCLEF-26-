from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
import numpy as np
import pandas as pd
import os
import librosa 
# data augumentations
from utils import DataPipelines, CFG, data_transforms
from sklearn.preprocessing import LabelEncoder
import torchaudio
class BirdDataset(Dataset):
    def __init__(self, df, sr=32000, augmentations=data_transforms(), data_dir=CFG.train_dir, apply_PCA=False):
        self.df = df
        self.data_dir = data_dir
        self.sr = sr
        self.augmentations = augmentations
        self.apply_PCA = apply_PCA
        
        # Otimização excelente! Declarar os transforms no __init__ economiza CPU.
        self.mel_spec_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sr,
            n_fft=1024,      
            hop_length=512,  
            n_mels=64        
        )
        self.mel_to_dB = torchaudio.transforms.AmplitudeToDB()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = os.path.join(self.data_dir, row['filename'])
        
        # Carrega o áudio
        waveform, sr = torchaudio.load(file_path)
        
        # Converte para mono se for estéreo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        max_length = 5 * self.sr # chunk 5 seconds of audio
        
        # Padroniza o tempo
        if waveform.shape[1] > max_length:
            waveform = waveform[:, :max_length]
        elif waveform.shape[1] < max_length:
            pad_amount = max_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
            
        # Gera o espectrograma
        mel_spec = self.mel_spec_transform(waveform)
        mel_spec = self.mel_to_dB(mel_spec)
        
        # O torchaudio já devolve [1, 64, tempo], mas garantimos a dimensão aqui
        if mel_spec.ndim == 2:
            mel_spec = mel_spec.unsqueeze(0)
            
        # Aplica o PCA (se ativo)
        if self.apply_PCA:
            mel_spec = DataPipelines().audioPCA(mel_spec)
            
            # IMPORTANTE: Se o seu audioPCA devolver um numpy array,
            # precisamos voltar para Tensor para a aumentação não quebrar
            if not isinstance(mel_spec, torch.Tensor):
                mel_spec = torch.tensor(mel_spec, dtype=torch.float32) 
            
            if mel_spec.ndim == 2:
                mel_spec = mel_spec.unsqueeze(0)

        # Aplica as aumentações (Apenas 1 vez)
        if self.augmentations is not None:
            mel_spec = self.augmentations(mel_spec)
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-6)
        # Gera e codifica a label
        label = row['primary_label']
        label = torch.tensor(label).long()
        return mel_spec, label

        