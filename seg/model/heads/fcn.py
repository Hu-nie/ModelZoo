import torch
import torch.nn as nn
import torch.nn.functional as F
from model.layers import ConvModule

class FCNHead(nn.Module):
    def __init__(self, c_in, c_out, num_classes):
        """
        c_in: 입력 채널 수 (Number of input channels)
        c_out: 출력 채널 수 (Number of output channels)
        num_classes: 클래스 수 (Number of classes)
        """
        super(FCNHead, self).__init__()
        self.conv1 = ConvModule(c_in, c_out, k=3, p=1)  # ConvModule 사용
        self.conv2 = nn.Conv2d(c_out, num_classes, k=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class FCN(nn.Module):
    def __init__(self, backbone, c_in, c_out, num_classes):
        """
        backbone: 백본 네트워크 (Backbone network)
        c_in: 입력 채널 수 (Number of input channels)
        c_out: 출력 채널 수 (Number of output channels)
        num_classes: 클래스 수 (Number of classes)
        """
        super(FCN, self).__init__()
        self.backbone = backbone
        self.head = FCNHead(c_in, c_out, num_classes)  # 수정된 FCNHead 사용

    def forward(self, x):
        input_shape = x.shape[-2:]
        x = self.backbone(x)
        x = self.head(x)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        return x
    
if __name__ == '__main__':
    from seg.model.backbones.resnet import ResNetBackbone


    x = torch.randn(2, 3, 224, 224)
    backbone = ResNetBackbone('resnet50')

    fcn = FCN(backbone, 2048, 256, 21)
    out = fcn(x)
    print(out.shape)