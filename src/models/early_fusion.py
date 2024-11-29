from torchvision import models
import torch.nn as nn

class EarlyFusionAlexNet(nn.Module):
    
    def __init__(self):
        """
        The input to the model needs to be [batch_size, 3, 10, 224, 224],
        since AlexNet takes that as the input.
        
        The forward pass returns one logit, so BCE should be used.
        """
        super().__init__()
        
        self.model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
        
            
        # Adjust the first convolutional layer to accept more channels
        self.model.features[0] = nn.Conv2d(
            in_channels=3 * 10,  # 3 channels × 10 frames
            out_channels=self.model.features[0].out_channels,
            kernel_size=self.model.features[0].kernel_size,
            stride=self.model.features[0].stride,
            padding=self.model.features[0].padding
        )
        
        # reinitialize the first layer
        nn.init.kaiming_normal_(self.model.features[0].weight, mode='fan_out', nonlinearity='relu')
        if self.model.features[0].bias is not None:
            nn.init.constant_(self.model.features[0].bias, 0)
        
        for param in self.model.features.parameters():
            param.requires_grad = False  # Freeze other layers
        for param in self.model.features[0].parameters():
            param.requires_grad = True  # Train the modified first layer
        
        # Adjust the classifier
        self.model.classifier[6] = nn.Linear(4096, 10)
    
    
    def forward(self, x):
        batch_size, channels, frames, height, width = x.shape
        assert (channels, frames, height, width) == (3, 10, 224, 224), f"Expected input shape [batch_size, 3, 10, 224, 224], but got {x.shape}"
        x = x.view(batch_size, channels * frames, height, width)
        
        return self.model(x)


class EarlyFusionAlexNet64(nn.Module):
    
    def __init__(self):
        """
        The input to the model needs to be [batch_size, 3, 10, 64, 64],
        since AlexNet takes that as the input.
        
        The forward pass returns one logit, so BCE should be used.
        """
        super().__init__()
        
        self.model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
        
            
        # Adjust the first convolutional layer to accept more channels
        self.model.features[0] = nn.Conv2d(
            in_channels=3 * 10,  # 3 channels × 10 frames
            out_channels=self.model.features[0].out_channels,
            kernel_size=self.model.features[0].kernel_size,
            stride=self.model.features[0].stride,
            padding=self.model.features[0].padding
        )
        
        # reinitialize the first layer
        nn.init.kaiming_normal_(self.model.features[0].weight, mode='fan_out', nonlinearity='relu')
        if self.model.features[0].bias is not None:
            nn.init.constant_(self.model.features[0].bias, 0)
        
        for param in self.model.features.parameters():
            param.requires_grad = False  # Freeze other layers
        for param in self.model.features[0].parameters():
            param.requires_grad = True  # Train the modified first layer
        
        self.model.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(9216, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, 10),
            )
    
    
    def forward(self, x):
        batch_size, channels, frames, height, width = x.shape
        assert (channels, frames, height, width) == (3, 10, 64, 64), f"Expected input shape [batch_size, 3, 10, 64, 64], but got {x.shape}"
        x = x.view(batch_size, channels * frames, height, width)
        
        return self.model(x)


if __name__ == "__main__":
    model = EarlyFusionAlexNet64()
    print(model)
    