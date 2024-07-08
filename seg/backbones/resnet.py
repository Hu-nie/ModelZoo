import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights, ResNet101_Weights

class ResNetBackbone(nn.Module):
    def __init__(self, resnet_type='resnet50'):
        super(ResNetBackbone, self).__init__()
        self.resnet_type = resnet_type
        self._load_resnet()
        
    def _load_resnet(self):
        if self.resnet_type == 'resnet50':
            resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        elif self.resnet_type == 'resnet101':
            resnet = models.resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(f"Unsupported ResNet type: {self.resnet_type}")
        
        self.features = nn.Sequential(*list(resnet.children())[:-2])

    def forward(self, x):
        return self.features(x)

if __name__ == '__main__':
    model = ResNetBackbone('resnet50')
    x = torch.zeros(1, 3, 512, 512)
    outs = model(x)
    print(outs.shape)
