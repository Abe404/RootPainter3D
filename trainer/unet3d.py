# pylint: disable=C0111, W0221, R0902
"""
U-Net architecture based on:
https://arxiv.org/abs/1505.04597
And modified to use Group Normalization
https://arxiv.org/abs/1803.08494
And then modified to use residual style connections.
With other alterations.

Copyright (C) 2019, 2020 Abraham George Smith

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
from torch import nn
import torch.nn.functional as F



def pad_to_next_largest_valid(size):
    """ return how much to pad the size to get it to the
        next valid size for the network input """
    min_input = 36
    assert size >= min_input
    # size should be a multiple of 16 above the min. 
    total_pad = (16 - (size - min_input)) % 16
    pad_start = total_pad // 2
    pad_end = total_pad - pad_start
    return pad_start, pad_end
 

def pad_to_valid_size(x):
    d_pad_start, d_pad_end = pad_to_next_largest_valid(x.shape[2])
    h_pad_start, h_pad_end = pad_to_next_largest_valid(x.shape[3])
    w_pad_start, w_pad_end = pad_to_next_largest_valid(x.shape[4])
    pad_settings = (w_pad_start, w_pad_end,
                    h_pad_start, h_pad_end,
                    d_pad_start, d_pad_end)
    x = F.pad(x, pad_settings)
    valid_sizes = sorted([36 + (x*16) for x in range(30)], reverse=True)
    assert x.shape[4] in valid_sizes, f'{x.shape}, {valid_sizes}'
    assert x.shape[3] in valid_sizes, f'{x.shape}, {valid_sizes}'
    assert x.shape[2] in valid_sizes, f'{x.shape}, {valid_sizes}'
    return x, pad_settings


def crop_away_padding(x, pad_settings):
    """ crop to remove any additional padding 
        added to the input by the 'pad_to_valid_size' method.
    """
    (w_pad_start, w_pad_end,
     h_pad_start, h_pad_end,
     d_pad_start, d_pad_end) = pad_settings
    
    d_pad_end = x.shape[2] - d_pad_end
    h_pad_end = x.shape[3] - h_pad_end
    w_pad_end = x.shape[4] - w_pad_end
    cropped_x = x[:, :, 
                  d_pad_start:d_pad_end,
                  h_pad_start:h_pad_end,
                  w_pad_start:w_pad_end]
    return cropped_x


class DownBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.pool = nn.MaxPool3d(2)
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, in_channels*2,
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.GroupNorm(32, in_channels*2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels*2, in_channels*2,
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.GroupNorm(32, in_channels*2)
        )
        self.conv1x1 = nn.Sequential(
            # down sample channels again.
            nn.Conv3d(in_channels*2, in_channels,
                      kernel_size=1, stride=1, bias=False)
        )

    def forward(self, x):
        out1 = self.pool(x)
        out2 = self.conv1(out1)
        out3 = self.conv2(out2)
        out4 = self.conv1x1(out3)
        return out4 + out1


def crop_tensor(tensor, target):
    """ Crop tensor to target size """
    _, _, tensor_depth, tensor_height, tensor_width = tensor.size()
    _, _, crop_depth, crop_height, crop_width = target.size()
    left = (tensor_width - crop_width) // 2
    top = (tensor_height - crop_height) // 2
    right = left + crop_width
    bottom = top + crop_height
    d_start = (tensor_depth - crop_depth) // 2
    d_end = d_start + crop_depth
    cropped_tensor = tensor[:, :, d_start:d_end, top: bottom, left: right]
    return cropped_tensor


class UpBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.ConvTranspose3d(in_channels, in_channels,
                               kernel_size=2, stride=2, padding=1),
            nn.ReLU(),
            nn.GroupNorm(32, in_channels)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels, in_channels,
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.GroupNorm(32, in_channels)
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(in_channels, in_channels,
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.GroupNorm(32, in_channels)
        )

    def forward(self, x, down_out):
        out = self.conv1(x)
        cropped = crop_tensor(down_out, out)
        out = cropped + out # residual
        out = self.conv2(out)
        out = self.conv3(out)
        return out



class UNet3D(nn.Module):
    def __init__(self, num_classes, im_channels=3):
        super().__init__()
        self.conv_in = nn.Sequential(
            nn.Conv3d(im_channels, 64, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.GroupNorm(32, 64),
            nn.Conv3d(64, 64, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.GroupNorm(32, 64)
        )
        self.down1 = DownBlock(64)
        self.down2 = DownBlock(64)
        self.down3 = DownBlock(64)
        self.down4 = DownBlock(64)
        self.up1 = UpBlock(64)
        self.up2 = UpBlock(64)
        self.up3 = UpBlock(64)
        self.up4 = UpBlock(64)
        self.conv_out = nn.Sequential(
            nn.Conv3d(64, 2 * num_classes, kernel_size=1, padding=0),
            nn.ReLU()
        )

    def forward(self, x):
        x, pad_settings = pad_to_valid_size(x)

        out1 = self.conv_in(x)
        out2 = self.down1(out1)
        out3 = self.down2(out2)
        out4 = self.down3(out3)
        out5 = self.down4(out4)
        out = self.up1(out5, out4)
        out = self.up2(out, out3)
        out = self.up3(out, out2)
        out = self.up4(out, out1)
        out = self.conv_out(out)


        out = crop_away_padding(out, pad_settings)
        return out
