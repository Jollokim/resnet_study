import torch

import torch.nn as nn

from timm.models.registry import register_model

__all__ = [
    'ResNetV1'
]


class PaddingSubBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        zeros = torch.zeros((x.shape[0], self.out_channels-self.in_channels, x.shape[-2], x.shape[-1])).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        x = torch.concat([x, zeros], dim=1)

        return x

class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, downsample: bool, shortcut: str='projection'):
        """
        shortcut:
            identity if n feature maps doesn't change
            projection if n feature map or size changes
            padding if n feature map or size changes
        """
        super().__init__()

        if downsample:
            self.conv1 = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1
            )

            
            if shortcut == 'projection':
                self.shortcut = nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=2,
                )
            # TODO: Implement padding, set up if
            elif shortcut == 'padding':
                pass
        else:
            self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding='same'
            )

        
            if in_channels < out_channels:
                # TODO: Implement padding, set up if
                if shortcut == 'projection':
                    self.shortcut = nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=1,
                        stride=1,
                        padding='same'
                    )
                elif shortcut == 'padding':
                    pass
                
            else:
                self.shortcut = nn.Identity()


        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding='same'
        )

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        init_x = x

        # print('init x', init_x.shape)
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        shortcut = self.shortcut(init_x)

        # print('conv x', x.shape)
        # print('shortcut', shortcut.shape)
        # print()

        x = x + shortcut
        
        x = torch.relu(x)

        return x





class ResNet(nn.Module):
    def __init__(self, n_classes: int) -> None:
        super().__init__()

        # define layers
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=7, # originally 7
                stride=2 # originally 2
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=3,
                stride=2 # originally 2
            ),
            ResBlock(64, 64, False),
            ResBlock(64, 64, False),

            ResBlock(64, 128, True),
            ResBlock(128, 128, False),

            ResBlock(128, 256, True),
            ResBlock(256, 256, False),

            ResBlock(256, 512, False),
            ResBlock(512, 512, False),

            nn.AvgPool2d(2, 2),
            nn.Flatten()
        )

        self.head = nn.Linear(2048, n_classes)
        # define output layer

    def forward(self, x):
        x = self.conv(x)
        x = self.head(x)
        x = torch.softmax(x, dim=1)
        return x


@register_model
def ResNetV1(pretrained=False, n_classes=200, **kwargs):
    model = ResNet(n_classes)

    return model

if __name__ == '__main__':
    from torchsummary import summary

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet(200).to(device)

    summary(model, input_size=(3, 64, 64))



    X = torch.randn((1, 3, 64, 64))