import torch
from torchvision import models
import torch.nn as nn

N_CLASSES = 10

class DummyCNN(nn.Module):
    def __init__(self):
        super(DummyCNN, self).__init__()
        self.convolutional = nn.Sequential(
            # Input Layer: 3x64x64
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            
            # Pooling layer: 16x32x32
            nn.MaxPool2d(2),
            
            # Layer 2: 16x32x32
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
        )
        
        self.fully_connected = nn.Sequential(
            nn.Linear(16*32*32, N_CLASSES),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        x = self.convolutional(x)
        x = x.view(x.size(0), -1)
        x = self.fully_connected(x)
        return x


class AlexNetFeatureExtractor(nn.Module):
    def __init__(self):
        super(AlexNetFeatureExtractor, self).__init__()
        # Load the pretrained AlexNet model
        alexnet = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
        # Keep only the features (convolutional layers)
        self.features = alexnet.features
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        # Forward pass through the convolutional layers
        x = self.features(x)
        return x
