import torch

import torch.nn as nn

from timm.models.registry import register_model

__all__ = [
    'ResNet56Projection'
]

# Inspired by the implementation of: https://towardsdev.com/implement-resnet-with-pytorch-a9fb40a77448
class ResBlockProjection(nn.Module):
    def __init__(self, in_channels, out_channels, downsample):
        super().__init__()
        if downsample:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.shortcut = nn.Sequential()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = x + shortcut

        return torch.relu(x)


class ResNet56(nn.Module):
    def __init__(self, in_channels, resblock, outputs=200):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding='same'),

            # res blocks
            resblock(32, 32, downsample=False),
            resblock(32, 32, downsample=False),
            resblock(32, 32, downsample=False),
            resblock(32, 32, downsample=False),
            resblock(32, 32, downsample=False),
            resblock(32, 32, downsample=False),
            resblock(32, 32, downsample=False),
            resblock(32, 32, downsample=False),
            resblock(32, 32, downsample=False),

            resblock(32, 64, downsample=True),
            resblock(64, 64, downsample=False),
            resblock(64, 64, downsample=False),
            resblock(64, 64, downsample=False),
            resblock(64, 64, downsample=False),
            resblock(64, 64, downsample=False),
            resblock(64, 64, downsample=False),
            resblock(64, 64, downsample=False),
            resblock(64, 64, downsample=False),

            resblock(64, 128, downsample=True),
            resblock(128, 128, downsample=False),
            resblock(128, 128, downsample=False),
            resblock(128, 128, downsample=False),
            resblock(128, 128, downsample=False),
            resblock(128, 128, downsample=False),
            resblock(128, 128, downsample=False),
            resblock(128, 128, downsample=False),
            resblock(128, 128, downsample=False),


            # pooling
            torch.nn.AdaptiveAvgPool2d(1),

            # flatten output
            nn.Flatten()

        )
        
        
        self.fc = torch.nn.Linear(128, outputs)

    def forward(self, x):

        x = self.conv(x)

        x = self.fc(x)

        torch.softmax(x, 1)

        return x

@register_model
def ResNet56Projection(pretrained=False, n_classes=200, **kwargs):
    model = ResNet56(3, ResBlockProjection)

    return model

if __name__ == '__main__':
    from torchsummary import summary

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet56(3, ResBlockProjection).to(device)

    summary(model, input_size=(3, 64, 64))