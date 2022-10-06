import torch

import torch.nn as nn

from timm.models.registry import register_model

__all__ = [
    'ResNet56ProjectionReLu',
    'ResNet56ProjectionSeLu',
    'ResNet56ProjectionLeakyReLu',
    'ResNet56PaddingReLu',
    'ResNet56PaddingSeLu',
    'ResNet56PaddingLeakyReLu'
]

# Design inspired by: https://towardsdev.com/implement-resnet-with-pytorch-a9fb40a77448
# However, this implementation doesn't use activation accordingly to the paper, this is fixed in this implementation
# We also modify for use of different activation functions
class ResBlockProjection(nn.Module):
    def __init__(self, in_channels, out_channels, downsample, activation):
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
        
        self.act1 = activation()
        self.act2 = activation()
        
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = x + shortcut

        return self.act2(x)

class PadBlock(nn.Module):
    def __init__(self, in_channels) -> None:
        super().__init__()
        self.in_channels = in_channels

    def forward(self, x):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pad = torch.zeros(x.shape).to(device)

        return torch.concat((x, pad), dim=1)

class ResBlockPadding(nn.Module):
    def __init__(self, in_channels, out_channels, downsample, activation):
        super().__init__()
        if downsample:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=2),
                nn.BatchNorm2d(in_channels),
                PadBlock(in_channels)
            )
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.shortcut = nn.Sequential()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
        self.act1 = activation()
        self.act2 = activation()
        
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = x + shortcut

        return self.act2(x)


class ResNet56(nn.Module):
    def __init__(self, in_channels, resblock, activation, outputs=200):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(32),
            activation(),

            # res blocks
            resblock(32, 32, downsample=False, activation=activation),
            resblock(32, 32, downsample=False, activation=activation),
            resblock(32, 32, downsample=False, activation=activation),
            resblock(32, 32, downsample=False, activation=activation),
            resblock(32, 32, downsample=False, activation=activation),
            resblock(32, 32, downsample=False, activation=activation),
            resblock(32, 32, downsample=False, activation=activation),
            resblock(32, 32, downsample=False, activation=activation),
            resblock(32, 32, downsample=False, activation=activation),

            resblock(32, 64, downsample=True, activation=activation),
            resblock(64, 64, downsample=False, activation=activation),
            resblock(64, 64, downsample=False, activation=activation),
            resblock(64, 64, downsample=False, activation=activation),
            resblock(64, 64, downsample=False, activation=activation),
            resblock(64, 64, downsample=False, activation=activation),
            resblock(64, 64, downsample=False, activation=activation),
            resblock(64, 64, downsample=False, activation=activation),
            resblock(64, 64, downsample=False, activation=activation),

            resblock(64, 128, downsample=True, activation=activation),
            resblock(128, 128, downsample=False, activation=activation),
            resblock(128, 128, downsample=False, activation=activation),
            resblock(128, 128, downsample=False, activation=activation),
            resblock(128, 128, downsample=False, activation=activation),
            resblock(128, 128, downsample=False, activation=activation),
            resblock(128, 128, downsample=False, activation=activation),
            resblock(128, 128, downsample=False, activation=activation),
            resblock(128, 128, downsample=False, activation=activation),

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
def ResNet56ProjectionReLu(pretrained=False, n_classes=200, **kwargs):
    model = ResNet56(3, ResBlockProjection, nn.ReLU)

    return model

@register_model
def ResNet56ProjectionSeLu(pretrained=False, n_classes=200, **kwargs):
    model = ResNet56(3, ResBlockProjection, nn.SELU)

    return model

@register_model
def ResNet56ProjectionLeakyReLu(pretrained=False, n_classes=200, **kwargs):
    model = ResNet56(3, ResBlockProjection, nn.LeakyReLU)

    return model

@register_model
def ResNet56PaddingReLu(pretrained=False, n_classes=200, **kwargs):
    model = ResNet56(3, ResBlockPadding, nn.ReLU)

    return model

@register_model
def ResNet56PaddingSeLu(pretrained=False, n_classes=200, **kwargs):
    model = ResNet56(3, ResBlockPadding, nn.SELU)

    return model

@register_model
def ResNet56PaddingLeakyReLu(pretrained=False, n_classes=200, **kwargs):
    model = ResNet56(3, ResBlockPadding, nn.LeakyReLu)

    return model

if __name__ == '__main__':
    from torchsummary import summary

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet56(3, ResBlockPadding, nn.ReLU).to(device)

    summary(model, input_size=(3, 64, 64))