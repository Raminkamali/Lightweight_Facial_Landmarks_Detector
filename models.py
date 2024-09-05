import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small

class KeyPointsModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone =  mobilenet_v3_small(weights="IMAGENET1K_V1").features
        self.dim_red = nn.Sequential(
            nn.Conv2d(in_channels=576,out_channels=128,kernel_size=3,padding=1),
            nn.InstanceNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128,out_channels=64,kernel_size=3,padding=1),
            nn.InstanceNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )

        self.head = nn.Linear(in_features=64,out_features=10,)
    def forward(self,x):
        x = self.dim_red(self.backbone(x))
        x = torch.mean(x,dim=[2,3])
        return self.head(x)
    