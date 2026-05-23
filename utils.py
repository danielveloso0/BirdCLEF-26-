import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
from sklearn.decomposition import PCA
import scipy.signal as signal
from IPython.display import Audio
from librosa.feature import melspectrogram
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import os
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
# Pytorch data augumentation
from torchvision import transforms
import random

class CFG():
    df_train = pd.read_csv('/kaggle/input/competitions/birdclef-2026/train.csv')
    train_soundscape =pd.read_csv('/kaggle/input/competitions/birdclef-2026/train_soundscapes_labels.csv') 
    train_dir = '/kaggle/input/competitions/birdclef-2026/train_audio'
    test_dir = '/kaggle/input/competitions/birdclef-2026/test_audio'
    num_class = len(df_train.primary_label.unique()) 
    SAMPLE_RATE = 32000
    N_MELS     = 64
    N_FFT      = 1024
    HOP_LENGTH = 512
class DataPipelines():
    def __init__(self,nPCA=1):
        self.nComponents = nPCA
    def open_audio(self,file_path, sr=CFG.SAMPLE_RATE,mode=None):
            
        # The audio format must be .OGG or similiar
        audio, sr = librosa.load(file_path)
       
        return audio, sr
    def add_noise(self,x, snr_db=15):
        """
        Adiciona ruído Gaussiano ao array de áudio 'x'.
        'snr_db' controla o quão alto o sinal original é em relação ao ruído.
        Valores entre 10 e 20 geralmente adicionam um chiado de fundo sem estragar o áudio.
        """
        # 1. Calcula a potência (energia) do sinal de áudio original
        signal_power = np.mean(x**2)
        
        # Prevenção: se o áudio for puro silêncio (potência zero), retorna o original
        if signal_power == 0:
            return x
    
        # 2. Calcula qual deve ser a potência do ruído para manter a proporção SNR desejada
        noise_power = signal_power / (10 ** (snr_db / 10))
        
        # 3. Gera o ruído branco gaussiano (média 0, desvio padrão baseado na potência)
        noise = np.random.normal(0, np.sqrt(noise_power), len(x))
        
        # 4. Soma o ruído ao áudio original
        noisy_x = x + noise
        
        return noisy_x
    
    def apply_bandpass_filter(self,y, sr, lowcut=1500, highcut=12000, order=5):
        """
        Aplica um filtro passa-faixa (band-pass) ao array de áudio.
        
        Parâmetros:
        y: array NumPy contendo o sinal de áudio (saída do librosa.load).
        sr: Taxa de amostragem (sample rate) do áudio.
        lowcut: Frequência mínima em Hz.
        highcut: Frequência máxima em Hz.
        order: Ordem do filtro.
        
        """
        # A frequência de Nyquist é sempre a metade da taxa de amostragem
        nyquist = 0.5 * sr
        
        # Normaliza as frequências para o padrão exigido pelo SciPy (entre 0 e 1)
        low = lowcut / nyquist
        high = highcut / nyquist
        # 1. Cria os coeficientes do filtro Butterworth passa-faixa ('band')
        b, a = signal.butter(order, [low, high], btype='band', analog=False)
        
        # 2. Aplica o filtro ao áudio usando filtfilt (fase zero para evitar atrasos)
        y_filtered = signal.filtfilt(b, a, y)
        
        return np.asfortranarray(y_filtered)
    def audioPCA(self, x, n=0.95, denoise=True):
        """
        Aplica PCA ao espectrograma para redução de dimensionalidade e denoising.
        
        Parâmetros:
        x: Array 2D do espectrograma no formato (n_mels, frames_de_tempo).
        n: Se inteiro, é o nº exato de componentes. Se for um float entre 0.0 e 1.0, 
           é a porcentagem da variância (informação) que o modelo deve manter (ex: 0.95).
        denoise: Se True, reconstrói o espectrograma limpo. Se False, retorna os PCs crus.
        """
        if n is None:
            n = self.nComponents
            
        # 1. Ajuste de Formato (O Pulo do Gato)
        # O espectrograma é (n_mels, tempo). O PCA do Scikit-Learn espera (amostras, features).
        # Ao transpor (.T), transformamos o Tempo em 'amostras' e as Frequências em 'features'.
        # Isso faz o PCA procurar os padrões de frequência mais marcantes no tempo (o canto).
        x_transposto = x.T 
        
        # 2. Inicializa o modelo PCA
        pca = PCA(n_components=n)
        
        # 3. Redução de Dimensionalidade (Aqui o ruído é separado da ave)
        PCs = pca.fit_transform(x_transposto)
        
        if denoise: 
            # O ruído, que ficou nos componentes descartados, simplesmente não volta.
            x_reconstruido = pca.inverse_transform(PCs)
            
            # Transpomos de volta para o formato de espectrograma (n_mels, frames_de_tempo)
            return x_reconstruido.T
            
        else:
            # Retorna apenas a matriz reduzida (útil para modelos clássicos como Random Forest)
            return PCs
    def mel_spectogram(self, x, n_mel=CFG.N_MELS, n_ffts=CFG.N_FFT, length=CFG.HOP_LENGTH):
        x = np.asarray(x).squeeze()
        mel_spec = melspectrogram(y=x, sr=CFG.SAMPLE_RATE, n_fft=n_ffts, 
            hop_length=length, 
            n_mels=n_mel)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec_db


def spec_augment(spec, num_masks=2, freq_mask_pct=0.15, time_mask_pct=0.15):
    """
    Aplica as técnicas de Frequency e Time Masking em um espectrograma.
    
    Parâmetros:
    spec: Array 2D do espectrograma no formato (n_mels, frames_de_tempo).
    num_masks: Quantas faixas apagar (2 costuma ser um bom número).
    freq_mask_pct: Porcentagem máxima (0 a 1) do eixo de frequência a ser apagada.
    time_mask_pct: Porcentagem máxima (0 a 1) do eixo de tempo a ser apagada.
    """
    # Cria uma cópia para não alterar o espectrograma original na memória
    aug_spec = spec.copy()
    n_mels, n_frames = aug_spec.shape
    
    # Preencher as faixas com a média do espectrograma é melhor do que com zero,
    # pois evita criar "degraus" ou bordas pretas muito severas e irreais.
    mean_val = aug_spec.mean()

    # 1. Frequency Masking (Corta faixas horizontais)
    for _ in range(num_masks):
        # Define a largura da faixa a ser cortada
        f_max = int(n_mels * freq_mask_pct)
        f_width = random.randint(0, f_max)
        
        # Define onde o corte começa
        f_start = random.randint(0, n_mels - f_width)
        
        # Aplica a máscara
        aug_spec[f_start:f_start + f_width, :] = mean_val

    # 2. Time Masking (Corta faixas verticais)
    for _ in range(num_masks):
        # Define a largura da faixa a ser cortada
        t_max = int(n_frames * time_mask_pct)
        t_width = random.randint(0, t_max)
        
        # Define onde o corte começa
        t_start = random.randint(0, n_frames - t_width)
        
        # Aplica a máscara
        aug_spec[:, t_start:t_start + t_width] = mean_val

    return aug_spec
def data_transforms(mode='train'):
    if mode == 'train':
        return transforms.Compose([
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    else:
        return transforms.Compose([
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    
def torchPCA(mel_spec, n_components=32, reconstruct=True):
    """
    Aplica PCA via SVD nativo do PyTorch em um espectrograma.
    
    Args:
        mel_spec (Tensor): Espectrograma no formato [1, Mels, Tempo] (ex: [1, 64, 313]).
        n_components (int): Número de componentes principais para manter.
        reconstruct (bool): Se True, reconstrói o espectrograma filtrado (mesmo tamanho original).
                            Se False, retorna o espectrograma reduzido [1, n_components, Tempo].
    Returns:
        Tensor: O espectrograma processado via PCA.
    """
    # 1. Remove a dimensão do canal para facilitar a matemática: vira [64, 313]
    x = mel_spec.squeeze(0) 
    
    # 2. O formato padrão para PCA é [Amostras, Features]. 
    # Vamos transpor de [Mels, Tempo] para [Tempo, Mels] (ex: [313, 64])
    x = x.T 
    
    # 3. Centralizar os dados (Subtrair a média de cada feature/Mel)
    mean = torch.mean(x, dim=0)
    x_centered = x - mean
    
    # 4. Aplicar o SVD (Decomposição em Valores Singulares)
    # U: Vetores singulares à esquerda
    # S: Valores singulares
    # Vh: Vetores singulares à direita (já transpostos no PyTorch)
    U, S, Vh = torch.linalg.svd(x_centered, full_matrices=False)
    
    # 5. Selecionar os top 'k' componentes principais (Truncamento)
    Vh_k = Vh[:n_components, :] # Formato: [k, Mels]
    
    # 6. Projetar os dados originais no novo espaço (Dimensionalidade Reduzida)
    x_projected = x_centered @ Vh_k.T # Formato: [Tempo, k]
    
    if reconstruct:
        # Reconstrói os dados originais (Denoising). Remove o ruído jogando fora componentes fracos.
        x_reconstructed = (x_projected @ Vh_k) + mean
        # Transpõe de volta para [Mels, Tempo] e devolve o canal no início
        return x_reconstructed.T.unsqueeze(0)
    else:
        # Se você quiser alimentar o modelo com o formato reduzido
        return x_projected.T.unsqueeze(0)
