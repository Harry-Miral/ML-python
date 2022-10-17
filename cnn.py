import torch.nn as nn
import os

savepath = r'features_whitegirl'
if not os.path.exists(savepath):
    os.mkdir(savepath)


class CnnNet(nn.Module):

    def __init__(self, num_classes=4):
        super(CnnNet, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=1, stride=2),

            nn.Conv2d(64, 192, kernel_size=3, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=1, stride=2),

            nn.Conv2d(192, 368, kernel_size=3, padding=1),
            nn.BatchNorm2d(368),
            nn.ReLU(inplace=True),

            nn.Conv2d(368, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((6, 6))
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(6 * 6 * 32, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.feature(x)
        feature = x.view(-1, 6 * 6 * 32)
        x = self.classifier(feature)
        return x, feature
