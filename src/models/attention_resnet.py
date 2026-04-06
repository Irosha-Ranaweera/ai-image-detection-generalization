import torch
import torch.nn as nn
from torchvision import models

from src.models.eca import ECA


class AttentionResNet(nn.Module):
    def __init__(self, model_name: str = "resnet18", num_classes: int = 2):
        super().__init__()

        if model_name == "resnet18":
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            out_channels = 512
        elif model_name == "resnet50":
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            out_channels = 2048
        else:
            raise ValueError("model_name must be 'resnet18' or 'resnet50'")

        for param in self.backbone.parameters():
            param.requires_grad = False

        self.eca = ECA(kernel_size=3)
        self.classifier = nn.Linear(out_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.eca(x)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x