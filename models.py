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
                self.backbone.conv1.weight.data = original_conv1.weight.data.sum(dim=1, keepdim=True)
            
            # 3. Substitui a última camada para focar nas espécies do Pantanal
            num_ftrs = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(num_ftrs, num_classes)
            
        else:
            raise ValueError(f"Backbone '{backbone}' not supported.")
            
    def forward(self, x):
        return self.backbone(x)

class BirdClassifierV2(nn.Module):
    def __init__(self, num_classes=CFG.num_class, backbone="resnet18", in_channels=1):
        super().__init__()

        self.backbone = self._create_backbone(backbone)

        self._adapt_first_conv(in_channels)
        self._adapt_classifier(backbone, num_classes)

    def _create_backbone(self, backbone):
        if backbone == "resnet18":
            return models.resnet18(weights="DEFAULT")

        if backbone == "resnet50":
            return models.resnet50(weights="DEFAULT")

        if backbone == "efficientnet_b0":
            return models.efficientnet_b0(weights="DEFAULT")

        if backbone == "mobilenet_v3_small":
            return models.mobilenet_v3_small(weights="DEFAULT")

        if backbone == "densenet121":
            return models.densenet121(weights="DEFAULT")

        raise ValueError(f"Backbone '{backbone}' not supported.")

    def _adapt_first_conv(self, in_channels):
        if in_channels == 3:
            return

        first_conv_name, first_conv = self._find_first_conv(self.backbone)

        new_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            dilation=first_conv.dilation,
            groups=first_conv.groups,
            bias=first_conv.bias is not None,
            padding_mode=first_conv.padding_mode,
        )

        if in_channels == 1 and first_conv.weight.shape[1] == 3:
            new_conv.weight.data = first_conv.weight.data.sum(dim=1, keepdim=True)
        else:
            nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")

        self._set_module_by_name(self.backbone, first_conv_name, new_conv)

    def _adapt_classifier(self, backbone, num_classes):
        if backbone.startswith("resnet"):
            num_ftrs = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(num_ftrs, num_classes)

        elif backbone.startswith("efficientnet"):
            num_ftrs = self.backbone.classifier[1].in_features
            self.backbone.classifier[1] = nn.Linear(num_ftrs, num_classes)

        elif backbone.startswith("mobilenet"):
            num_ftrs = self.backbone.classifier[3].in_features
            self.backbone.classifier[3] = nn.Linear(num_ftrs, num_classes)

        elif backbone.startswith("densenet"):
            num_ftrs = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Linear(num_ftrs, num_classes)

        else:
            raise ValueError(f"Classifier adaptation not implemented for '{backbone}'.")

    def _find_first_conv(self, module):
        for name, child in module.named_modules():
            if isinstance(child, nn.Conv2d):
                return name, child

        raise ValueError("No Conv2d layer found in backbone.")

    def _set_module_by_name(self, model, name, new_module):
        parts = name.split(".")
        parent = model

        for part in parts[:-1]:
            parent = getattr(parent, part)

        setattr(parent, parts[-1], new_module)

    def forward(self, x):
        return self.backbone(x)    