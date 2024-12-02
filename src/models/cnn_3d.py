import torch
from torch.nn import LazyConv3d, LazyLinear
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models.video import s3d, mc3_18, r3d_18, mvit_v2_s
# s3d: """Construct Separable 3D CNN model.
    #Reference: `Rethinking Spatiotemporal Feature Learning <https://arxiv.org/abs/1712.04851>`__.
# mc3_18: <resnet module>"""Construct 18 layer Mixed Convolution network as in
        #Reference: `A Closer Look at Spatiotemporal Convolutions for Action Recognition <https://arxiv.org/abs/1711.11248>`__.
#r3d_18: <resnet module> """Construct 18 layer Resnet3D model.
        #Reference: `A Closer Look at Spatiotemporal Convolutions for Action Recognition <https://arxiv.org/abs/1711.11248>`__.
#mvit_v2_s:  <swin_transformer module>"""Constructs a Vision Transformer model based on the ViT architecture.
         #     `Multiscale Vision Transformers <https://arxiv.org/abs/2104.11227>`__ and
        #     `MViTv2: Improved Multiscale Vision Transformers for Classification
        #     and Detection <https://arxiv.org/abs/2112.01526>`__.

from typing import Any, Callable, List, Optional, Sequence, Tuple, Type, Union
from torch import Tensor

device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class Block3D(nn.Module):
    def __init__(self, ch):
        super(Block3D, self).__init__()
        self.block= nn.Sequential(
            nn.LazyBatchNorm3d(),
            nn.Dropout3d(p=0.1),
            LazyConv3d(out_channels=ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.LazyBatchNorm3d(),
            nn.Dropout3d(p=0.1),
            LazyConv3d(out_channels=ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            
            nn.LazyBatchNorm3d(),
            nn.Dropout3d(p=0.1),
            LazyConv3d(out_channels=ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.LazyBatchNorm3d(),

            )
    def forward(self, x):
        x= self.block(x)
        return x

class Basic3DCNN(nn.Module):
    def __init__(self, num_classes=10, ch=64, depth=8):
        super(Basic3DCNN, self).__init__()
        self.num_classes = num_classes
        self.initblock= Block3D(ch)
        self.blocks = nn.ModuleList([Block3D(ch) for _ in range(depth-1)])
        self.classifier= nn.Sequential(
            nn.AdaptiveAvgPool3d((10,2,2)),
            nn.Flatten(),
            nn.Linear(in_features=40*ch,out_features=num_classes)
        )
        self.time_embed = nn.Embedding(10, 3)


        
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, channels, num_frames, height, width)
        Returns:
            logits: Tensor of shape (batch_size, num_classes)
        """
        # Add time embedding
        batch_size, channels, num_frames, height, width = x.shape
        # time_embed = torch.arange(num_frames, device=x.device)
        # time_embed = self.time_embed(time_embed)[:,:,None,None,None]
        # time_embed = time_embed.permute(2, 1, 0, 3, 4)
        # x = x + time_embed
        x= self.initblock(x)
        for block in self.blocks:
            x= x + block(x)
        x= self.classifier(x)
        return x
        

class Pretrained3dCNN(nn.Module):
    # https://arxiv.org/abs/1711.11248
    def __init__(self, num_classes=10):
        super(Pretrained3dCNN, self).__init__()
        
        # Replace the final fully connected layer to match the number of classes
        # self.model = models.video.mc3_18(pretrained=True)
        # in_features = self.model.fc.in_features
        # self.model.fc = nn.Linear(in_features, num_classes)
        # Load a pretrained S3D model
        self.model = r3d_18(pretrained=True)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, channels, num_frames, height, width)
        Returns:
            logits: Tensor of shape (batch_size, num_classes)
        """
        return self.model(x)
    
class PretrainedVisionTransformer(nn.Module):
    # https://arxiv.org/abs/1711.11248
    def __init__(self, num_classes=10):
        super(PretrainedVisionTransformer, self).__init__()
        
        # Replace the final fully connected layer to match the number of classes
        # self.model = models.video.mc3_18(pretrained=True)
        # in_features = self.model.fc.in_features
        # self.model.fc = nn.Linear(in_features, num_classes)
        # Load a pretrained S3D model
        self.model = mvit_v2_s(pretrained=True)
        self.model.head = nn.LazyLinear(num_classes)
        
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, channels, num_frames, height, width)
        Returns:
            logits: Tensor of shape (batch_size, num_classes)
        """
        return self.model(x)

if __name__ == "__main__":
    model = Basic3DCNN(num_classes=10)
    print(model)