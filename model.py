"""ResNet model definition for image classification.

This module provides a PyTorch ResNet model that works out of the box.
"""

import torch
import torch.nn as nn
from torchvision import models


class ResNetModel(nn.Module):
    """ResNet model for image classification.

    Supports pretrained weights and custom number of classes.
    """

    def __init__(self, architecture="resnet18", num_classes=10, pretrained=True, dropout=0.0):
        """Initialize ResNet model.

        Args:
            architecture: ResNet variant (resnet18, resnet34, resnet50, etc.)
            num_classes: Number of output classes
            pretrained: Whether to use ImageNet pretrained weights
            dropout: Dropout probability (0.0 = no dropout)
        """
        super(ResNetModel, self).__init__()

        # Load pretrained ResNet
        if architecture == "resnet18":
            self.backbone = models.resnet18(pretrained=pretrained)
        elif architecture == "resnet34":
            self.backbone = models.resnet34(pretrained=pretrained)
        elif architecture == "resnet50":
            self.backbone = models.resnet50(pretrained=pretrained)
        elif architecture == "resnet101":
            self.backbone = models.resnet101(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")

        # Get number of features from the original FC layer
        num_features = self.backbone.fc.in_features

        # Replace final FC layer with custom classifier
        if dropout > 0:
            self.backbone.fc = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(num_features, num_classes)
            )
        else:
            self.backbone.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, 3, H, W)

        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        return self.backbone(x)


def create_model(config):
    """Create model from configuration.

    Args:
        config: Configuration dictionary with model parameters

    Returns:
        ResNetModel instance
    """
    model_config = config.get('model', {})

    model = ResNetModel(
        architecture=model_config.get('architecture', 'resnet18'),
        num_classes=model_config.get('num_classes', 10),
        pretrained=model_config.get('pretrained', True),
        dropout=model_config.get('dropout', 0.0)
    )

    return model
