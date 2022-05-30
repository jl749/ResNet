from typing import List

import torch
import torch.nn as nn


class Block(nn.Module):

    def __init__(self, c1: int, c2: int, identity_downsample: nn.Sequential = None, stride=1):
        super(Block, self).__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(
            c1, c2, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn1 = nn.BatchNorm2d(c2)
        self.conv2 = nn.Conv2d(  # preserve channels but reduce spatial dim
            c2,
            c2,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(c2)
        self.conv3 = nn.Conv2d(
            c2,
            c2 * self.expansion,  # point-wise convolution that increases the channel counts
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.bn3 = nn.BatchNorm2d(c2 * self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        self.stride = stride

    def forward(self, x):
        identity = x.clone()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:  # ResNet50,101,152 increase the channel at the end, match increased channels via point-wise
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, block: nn.Module, layers: List[int], image_channels: int, num_classes: int):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)  # halve spatial dim (224 --> 112), padding='SAME'
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # halve spatial dim again (112 --> 56), padding='SAME'

        """
        e.g. ResNet50, 101, 152 example
        (batch, 56, 56, 64) --> layer1 --> (batch, 56, 56, 256)
        (batch, 56, 56, 256) --> layer2 --> (batch, 28, 28, 1024)
        (batch, 28, 28, 1024) --> layer3 --> (batch, 14, 14, 2048)
        """
        # ResNet18,34: each block produces 64 128 256 512 channels
        # ResNet50,101,152: each block produces 256 512 1024 2048 channels
        self.layer1 = self._make_layer(
            block, layers[0], intermediate_channels=64, stride=1
        )  # (takes 56, 56)
        self.layer2 = self._make_layer(
            block, layers[1], intermediate_channels=128, stride=2
        )  # (takes 28, 28)
        self.layer3 = self._make_layer(
            block, layers[2], intermediate_channels=256, stride=2
        )  # (takes 14, 14)
        self.layer4 = self._make_layer(
            block, layers[3], intermediate_channels=512, stride=2
        )  # (takes 7, 7)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

    def _make_layer(self, block, num_residual_blocks, intermediate_channels, stride):
        identity_downsample = None
        layers = []

        # Either if we half the input space for ex, 56x56 -> 28x28 (stride=2), or channels changes
        # we need to adapt the Identity (skip connection) so it will be able to be added
        # to the layer that's ahead
        if stride != 1 or self.in_channels != intermediate_channels * 4:
            identity_downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    intermediate_channels * 4,
                    kernel_size=1,  # point-wise
                    stride=stride,  # always 1 or 2
                    bias=False
                ),
                nn.BatchNorm2d(intermediate_channels * 4),
            )

        layers.append(
            block(self.in_channels, intermediate_channels, identity_downsample, stride)
        )

        # The expansion size is always 4 for ResNet 50,101,152
        self.in_channels = intermediate_channels * 4  # increased channel

        # For example for first resnet layer: 256 will be mapped to 64 as intermediate layer,
        # then finally back to 256. Hence no identity downsample is needed, since stride = 1,
        # and also same amount of channels.
        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, intermediate_channels))  # bottleneck layers

        return nn.Sequential(*layers)


def ResNet50(img_channel=3, num_classes=1000):
    return ResNet(Block, [3, 4, 6, 3], img_channel, num_classes)


def ResNet101(img_channel=3, num_classes=1000):
    return ResNet(Block, [3, 4, 23, 3], img_channel, num_classes)


def ResNet152(img_channel=3, num_classes=1000):
    return ResNet(Block, [3, 8, 36, 3], img_channel, num_classes)


def test():
    net = ResNet50(img_channel=3, num_classes=1000)
    y = net(torch.randn(4, 3, 224, 224))
    print(y.size())


test()
