from torch import nn

class ConvModule(nn.Sequential):
    def __init__(self, c_in, c_out, k, s=1, p=0, d=1, g=1):
        """
        c_in: 입력 채널 수 (Number of input channels)
        c_out: 출력 채널 수 (Number of output channels)
        k: 커널 크기 (Kernel size)
        s: 스트라이드 (Stride, default=1)
        p: 패딩 (Padding, default=0)
        d: 커널의 공간적 확장 (Dilation, default=1)
        g: 그룹 수 (Number of groups, default=1)
        """
        super().__init__(
            nn.Conv2d(c_in, c_out, k, s, p, d, g, bias=False),  # 합성곱 층 (Convolution layer)
            nn.BatchNorm2d(c_out),  # 배치 정규화 (Batch normalization)
            nn.ReLU(inplace=True)  # 활성화 함수 (Activation function, ReLU)
        )