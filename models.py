import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models

### NETWORKS ###


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10, config=[32, 64]):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, config[0], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(config[0], config[1], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Linear(config[1] * 7 * 7, num_classes)

    def forward(self, x):
        out = self.features(x)
        out = torch.flatten(out, 1)
        return self.classifier(out)


class MyCNN(nn.Module):
    def __init__(self, num_classes=10, config=[32, 64, 128]):
        super().__init__()
        self.features1 = SingleConvBlock(1, config[0])
        self.features2 = SingleConvBlock(config[0], config[1])
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(config[1] * 7 * 7, config[2]),
            nn.ReLU(inplace=True),
            nn.Linear(config[2], num_classes))

    def forward(self, x):
        out = F.max_pool2d(self.features1(x), 2)
        out = F.max_pool2d(self.features2(out), 2)
        out = torch.flatten(out, 1)
        return self.classifier(out)


class MyResNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = SingleConvBlock(1, 64)

        resnet = models.resnet18()
        self.resnet_block = nn.Sequential(
            resnet.layer1,
            resnet.layer2,
            resnet.layer3
        )

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.stem(x)

        x = self.resnet_block(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return self.fc(x)


### BLOCKS OF LAYERS ###


class SingleConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(SingleConvBlock, self).__init__()

        self.conv = nn.Sequential(nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1),
                                  nn.BatchNorm2d(ch_out),
                                  nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)
