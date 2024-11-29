from torchvision import models
import torch.nn as nn
import torch

class TwoStreamAlexNet(nn.Module):
    def __init__(self):
        """
        The input to the model needs to be [batch_size, 227, 224, 224].
        First 3 channels are for spatial stream network.
        Other channels are for temporal stream network.
        """
        super().__init__()

        ### Initialize Spatial Stream Network ###
        self.spatial_stream = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
        
        # Adjust the first convolutional layer to accept more channels
        self.spatial_stream.features[0] = nn.Conv2d(
            in_channels=3,  # 3 image channels
            out_channels=self.spatial_stream.features[0].out_channels,
            kernel_size=self.spatial_stream.features[0].kernel_size,
            stride=self.spatial_stream.features[0].stride,
            padding=self.spatial_stream.features[0].padding
        )
        
        # reinitialize the first layer
        nn.init.kaiming_normal_(self.spatial_stream.features[0].weight, mode='fan_out', nonlinearity='relu')
        if self.spatial_stream.features[0].bias is not None:
            nn.init.constant_(self.spatial_stream.features[0].bias, 0)
        
        for param in self.spatial_stream.features.parameters():
            param.requires_grad = False  # Freeze other layers
        for param in self.spatial_stream.features[0].parameters():
            param.requires_grad = True  # Train the modified first layer
        
        # Adjust the classifier
        self.spatial_stream.classifier[6] = nn.Linear(4096, 256)

        ### Initialize Temporal Stream Network ###
        self.temporal_stream = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
        
        # Adjust the first convolutional layer to accept more channels
        self.temporal_stream.features[0] = nn.Conv2d(
            in_channels=224,  # 224 flow channels
            out_channels=self.temporal_stream.features[0].out_channels,
            kernel_size=self.temporal_stream.features[0].kernel_size,
            stride=self.temporal_stream.features[0].stride,
            padding=self.temporal_stream.features[0].padding
        )
        
        # reinitialize the first layer
        nn.init.kaiming_normal_(self.temporal_stream.features[0].weight, mode='fan_out', nonlinearity='relu')
        if self.temporal_stream.features[0].bias is not None:
            nn.init.constant_(self.temporal_stream.features[0].bias, 0)
        
        for param in self.temporal_stream.features.parameters():
            param.requires_grad = False  # Freeze other layers
        for param in self.temporal_stream.features[0].parameters():
            param.requires_grad = True  # Train the modified first layer
        
        # Adjust the classifier
        self.temporal_stream.classifier[6] = nn.Linear(4096, 256)

        ### Late fusion linear layer ###
        self.late_fusion = nn.Linear(512,10)
        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        assert (channels, height, width) == (227, 224, 224)
        x_spatial = x[:,0:3]
        x_temporal = x[:,3:228]
        y_spatial = self.relu(self.spatial_stream(x_spatial))
        y_temporal = self.relu(self.temporal_stream(x_temporal))
        y = torch.concat([y_spatial, y_temporal], dim=1)
        y = self.late_fusion(y)
        return y
    
if __name__ == "__main__":
    model = TwoStreamAlexNet()
    print(model)