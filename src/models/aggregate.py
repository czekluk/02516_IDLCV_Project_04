import torch
from torchvision import models
import torch.nn as nn

from models.basic_models import DummyCNN, AlexNetFeatureExtractor

class AggregateDummyNet(nn.Module):
    def __init__(self, num_classes=10, num_frames=10):
        """Baseline model for aggregating frame-level features using mean pooling. Uses a dummy CNN for feature extraction.

        Args:
            num_classes (int, optional): Number of predicted class types. Defaults to 10.
            num_frames (int, optional): Number of frames for a video. Defaults to 10.
        """
        super(AggregateDummyNet, self).__init__()
        self.num_frames = num_frames
        self.cnn_model = DummyCNN()

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, channels, num_frames, height, width)
        Returns:
            logits: Tensor of shape (batch_size, num_classes)
        """
        assert(x.shape[1:] == (3, 10, 64, 64)), f"Expected input shape [batch_size, 3, 10, 64, 64], but got {x.shape}"
        batch_size, channels, num_frames, height, width = x.shape
        
        # Reshape to combine batch_size and num_frames for efficient parallel processing
        x = x.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, channels, height, width) # Shape: [batch_size*num_frames, channels, height, width]
        output = self.cnn_model(x) # Shape: [batch_size*num_frames, num_classes]

        # Reshape back to separate videos and frames, aggregate frames using mean
        video_features = output.view(batch_size, num_frames, -1)  # Shape: [batch_size, num_frames, num_classes]
        video_features = video_features.mean(dim=1) # Shape: [batch_size, num_classes]

        return video_features

class AggregateAlexNet(nn.Module):
    def __init__(self, num_classes=10, num_frames=10):
        super(AggregateAlexNet, self).__init__()
        self.num_frames = num_frames
        self.cnn_model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
        # Freeze earlier layers
        for param in self.cnn_model.features.parameters():
            param.requires_grad = False
        
        # Adjust the classifier for background and pothole
        self.cnn_model.classifier[6] = nn.Linear(4096, num_classes)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, channels, num_frames, height, width)
        Returns:
            logits: Tensor of shape (batch_size, num_classes)
        """
        assert(x.shape[1:] == (3, 10, 224, 224)), f"Expected input shape [batch_size, 3, 10, 224, 224], but got {x.shape}"
        batch_size, channels, num_frames, height, width = x.shape
        
        # Reshape to combine batch_size and num_frames for efficient parallel processing
        x = x.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, channels, height, width) # Shape: [batch_size*num_frames, channels, height, width]
        output = self.cnn_model(x) # Shape: [batch_size*num_frames, num_classes]

        # Reshape back to separate videos and frames, aggregate frames using mean
        video_features = output.view(batch_size, num_frames, -1)  # Shape: [batch_size, num_frames, num_classes]
        video_features = video_features.mean(dim=1) # Shape: [batch_size, num_classes]

        return video_features
    
        # batch_size, channels, num_frames, height, width = x.shape
        
        # # Reshape to combine batch_size and num_frames for efficient parallel processing
        # x = x.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, channels, height, width)

        # # Extract frame-level features for all frames in parallel
        # frame_features = self.cnn_model(x)  # Shape: [batch_size * num_frames, 256, 6, 6]
        
        # # Reshape back to separate videos and frames
        # frame_features = frame_features.view(batch_size, num_frames, -1)  # Shape: [batch_size, num_frames, 256*6*6]

        # # Flatten features from all frames for each video
        # video_features = frame_features.reshape(batch_size, -1)  # Shape: [batch_size, num_frames * 256 * 6 * 6]

        # # Classify each video based on the combined features
        # logits = self.classifier(video_features)  # Shape: [batch_size, num_classes]

        # return logits
