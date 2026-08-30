"""Model definitions for Fashion-MNIST classification."""
import torch.nn as nn
from torchvision.models import resnet18


class SimpleCNN(nn.Module):
    """Small CNN sized for 28x28 grayscale input."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 28 -> 14
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 14 -> 7
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def _resnet18_grayscale(num_classes: int) -> nn.Module:
    """ResNet-18 adapted for single-channel 28x28 input."""
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def get_model(architecture: str, num_classes: int = 10) -> nn.Module:
    architecture = architecture.lower()
    if architecture == "cnn":
        return SimpleCNN(num_classes=num_classes)
    if architecture == "resnet18":
        return _resnet18_grayscale(num_classes=num_classes)
    raise ValueError(f"Unknown architecture: {architecture}")
