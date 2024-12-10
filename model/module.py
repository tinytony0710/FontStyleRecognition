import torch
import torch.nn as nn
import torch.nn.functional as F

## a convolutional kernel size of 3×3. The stride parameter is 2, and the padding parameter is "same".
#                                           ^what stride parameter??
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, device, dtype = torch.float32):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 3,
                padding = 'same', padding_mode = 'replicate',
                bias=True, device=device, dtype=dtype),
            nn.BatchNorm2d(out_channels, device=device, dtype=dtype),
            nn.ReLU(),
        )

    def forward(self, _x):
        output = self.layers(_x)
        return output

## These three skip connections add the output of the 3rd, 8th, and 12th layers as additional input for the following three layers.
class ResBlock(nn.Module):
    def __init__(self, channel_size, layer_count, device, dtype = torch.float32):
        super().__init__()
        self.layers = nn.Sequential()
        for i in range(layer_count-1):
            self.layers.append(ConvBlock(channel_size, channel_size, device, dtype))

        self.layers.append(nn.Conv2d(channel_size, channel_size, kernel_size = 3,
            padding = 'same', padding_mode = 'replicate',
            bias=True, device=device, dtype=dtype))
        self.layers.append(nn.BatchNorm2d(channel_size, device=device, dtype=dtype))

    def forward(self, _x):
        skip = _x
        output = self.layers(_x)
        output += skip
        output = F.relu(output)
        return output

class SwordNet(nn.Module):
    def __init__(self, prediction_class, device, dtype = torch.float32):
        super().__init__()

        # input (n,1,img_height=96,img_width=96)
        self.layers = nn.ModuleList([                           # (n,1,96,96)
            ConvBlock(1, 64, device=device, dtype=dtype),       # (n,64,96,96)
            ConvBlock(64, 128, device=device, dtype=dtype),     # (n,128,96,96)
            nn.MaxPool2d(kernel_size=2, stride=2),              # (n,128,48,48)
            ResBlock(128, 3, device=device, dtype=dtype),       # (n,128,48,48)
            nn.MaxPool2d(kernel_size=2, stride=2),              # (n,128,24,24)
            ConvBlock(128, 256, device=device, dtype=dtype),    # (n,256,24,24)
            ConvBlock(256, 256, device=device, dtype=dtype),    # (n,256,24,24)
            ResBlock(256, 3, device=device, dtype=dtype),       # (n,256,24,24)
            nn.MaxPool2d(kernel_size=2, stride=2),              # (n,256,12,12)
            ConvBlock(256, 512, device=device, dtype=dtype),    # (n,512,12,12)
            ResBlock(512, 3, device=device, dtype=dtype),       # (n,512,12,12)
            nn.MaxPool2d(kernel_size=2, stride=2),              # (n,512,6,6)
            ConvBlock(512, 1024, device=device, dtype=dtype),   # (n,1024,6,6)
            nn.AdaptiveMaxPool2d((1, 1)),                       # (n,1024,1,1)
            nn.Flatten(),                                       # (n,1024)
            nn.Dropout(p=0.5),                                  # (n,1024)
            nn.Linear(1024, 9, bias=True, device=device, dtype=dtype),
            # nn.Linear(729, 81, bias=True, device=device, dtype=dtype),
            # nn.Linear(36864, 81, bias=True, device=device, dtype=dtype),
            # nn.Linear(81, 9, bias=True, device=device, dtype=dtype),
            nn.Linear(9, prediction_class, bias=True, device=device, dtype=dtype),
            nn.Softmax(dim=-1),
        ])

    def forward(self, _x):
        output = _x
        for i, layer in enumerate(self.layers):
            # print(i)
            output = layer(output)
        return output
