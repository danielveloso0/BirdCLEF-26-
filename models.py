import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from utils import CFG
import torch.nn as nn
import torchvision.models as models

class BirdClassifier(nn.Module):
    def __init__(self, num_classes=CFG.num_class, backbone='resnet18', in_channels=1):
        super(BirdClassifier, self).__init__()
        
        if backbone == 'resnet18':
            # 1. Carrega os pesos de forma moderna
            self.backbone = models.resnet18(weights='DEFAULT')
            
            # 2. Ajuste para aceitar Espectrogramas de 1 canal (Grayscale)
            if in_channels == 1:
                # Pegamos as configurações originais da primeira camada
                original_conv1 = self.backbone.conv1
                # Criamos uma nova camada idêntica, mas que aceita apenas 1 canal
                self.backbone.conv1 = nn.Conv2d(
                    1, 64, kernel_size=7, stride=2, padding=3, bias=False
                )
                # Dica de ouro: Somamos os pesos originais do RGB para não perder o pré-treino
                self.backbone.conv1.weight.data = original_conv1.weight.data.sum(dim=1, keepdim=True)
            
            # 3. Substitui a última camada para focar nas espécies do Pantanal
            num_ftrs = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(num_ftrs, num_classes)
            
        else:
            raise ValueError(f"Backbone '{backbone}' not supported.")
            
    def forward(self, x):
        return self.backbone(x)

    