import torch.nn as nn
from torchvision import models

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


class LateFusionAlexNet(nn.Module):
    def __init__(self, num_classes=10, num_frames=10):
        super(LateFusionAlexNet, self).__init__()
        self.num_frames = num_frames
        self.cnn_model = AlexNetFeatureExtractor()  # CNN for extracting frame-level features
        self.classifier = nn.Sequential(
            nn.Linear(256 * 6 * 6 * num_frames, 4096),  # Flattened features from all frames
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, channels, num_frames, height, width)
        Returns:
            logits: Tensor of shape (batch_size, num_classes)
        """
        batch_size, channels, num_frames, height, width = x.shape
        
        # Reshape to combine batch_size and num_frames for efficient parallel processing
        x = x.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, channels, height, width)

        # Extract frame-level features for all frames in parallel
        frame_features = self.cnn_model(x)  # Shape: [batch_size * num_frames, 256, 6, 6]
        
        # Reshape back to separate videos and frames
        frame_features = frame_features.view(batch_size, num_frames, -1)  # Shape: [batch_size, num_frames, 256*6*6]

        # Flatten features from all frames for each video
        video_features = frame_features.reshape(batch_size, -1)  # Shape: [batch_size, num_frames * 256 * 6 * 6]

        # Classify each video based on the combined features
        logits = self.classifier(video_features)  # Shape: [batch_size, num_classes]

        return logits
