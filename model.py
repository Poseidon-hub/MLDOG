import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(num_classes: int, pretrained: bool) -> nn.Module:
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)

    if pretrained:
        for parameter in model.parameters():
            parameter.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
