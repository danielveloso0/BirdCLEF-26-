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
class BirdDataset(Dataset):
    def __init__(self, df, sr=32000, augmentations=data_transforms(), data_dir=CFG.train_dir, apply_PCA=False):
        le = LabelEncoder()
        df['primary_label'] = le.fit_transform(df['primary_label'])
        self.df = df
        self.data_dir = data_dir
        self.sr = sr
        self.augmentations = augmentations
        self.apply_PCA = apply_PCA
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.loc[idx]
        file_path = os.path.join(self.data_dir, row['filename'])
        audio, sr = DataPipelines().open_audio(file_path, sr=self.sr)
        mel_spec = DataPipelines().mel_spectogram(audio)  
        label = row['primary_label']
        # one hot encode the label 
        label = torch.nn.functional.one_hot(torch.tensor(label), num_classes=CFG.num_class)

        if self.apply_PCA:
            mel_spec = DataPipelines().audioPCA(mel_spec)
            audio = self.augmentations(mel_spec)
        if self.augmentations is not None:
            audio = self.augmentations(mel_spec)
        return mel_spec, label
        
        