import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, n_iteration=1):
        super(ConvBlock, self).__init__()

        self.convs = []
        middle_ch = in_ch
        for i in range(n_iteration):
            self.convs.append(nn.Conv2d(in_channels=middle_ch,
                                        out_channels=out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1))
            middle_ch = out_ch
        self.activation = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, inputs):
        for conv in self.convs:
            x = conv(inputs)
            x = self.activation(x)

        return self.pool(x)


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()

        self.Block1 = ConvBlock(3, 64)
        self.Block2 = ConvBlock(64, 128)
        self.Block3 = ConvBlock(128, 256)
        self.Block4 = ConvBlock(256, 512)
        self.Block5 = ConvBlock(512, 512)

    def forward(self, in_img):
        x = self.Block1(in_img)
        x = self.Block2(x)
        x = self.Block3(x)
        feature1 = x
        x = self.Block4(x)
        feature2 = x
        x = self.Block5(x)
        feature3 = x

        return feature1, feature2, feature3


def main():
    in_tensor = torch.rand(1, 3, 224, 224)
    print(f'In_tensor shape : {in_tensor.shape}')

    backbone = VGG16()
    features = backbone(in_tensor)
    print(f'Feature1 shape : {features[0].shape}')
    print(f'Feature2 shape : {features[1].shape}')
    print(f'Feature3 shape : {features[2].shape}')


if __name__ == "__main__":
    main()
