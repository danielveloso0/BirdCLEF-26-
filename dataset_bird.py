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
        le = LabelEncoder()
        df['primary_label'] = le.fit_transform(df['primary_label'])
        self.df = df
        self.data_dir = data_dir
        self.sr = sr
        self.augmentations = augmentations
        self.apply_PCA = apply_PCA
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
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
        waveform, sr = torchaudio.load(file_path)
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        max_length = 5 * self.sr # chunk 5 seconds of audio
        
        if waveform.shape[1] > max_length:
            # Corta se for maior
            waveform = waveform[:, :max_length]
        elif waveform.shape[1] < max_length:
            # Preenche com zeros se for menor
            pad_amount = max_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
        if mel_spec.ndim == 2:
            mel_spec = mel_spec.unsqueeze(0)
        label = row['primary_label']
        # one hot encode the label 
        label = torch.nn.functional.one_hot(torch.tensor(label), num_classes=CFG.num_class)
        mel_spec = self.mel_spec(waveform)
        mel_spec = self.mel_to_dB(mel_spec)
        if mel_spec.ndim == 2:
            mel_spec = mel_spec.unsqueeze(0)
        if self.apply_PCA:
            mel_spec = DataPipelines().audioPCA(mel_spec)
            mel_spec = self.augmentations(mel_spec)
        if self.augmentations is not None:
            mel_spec = self.augmentations(mel_spec)
        return mel_spec, label

        