# This file contains the functions to load different data modalities from hdf5
# Copyright (c) 2021 Robert Bosch GmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# This source code is derived from Autonomous Vision - Occupancy Networks
#   (https://github.com/autonomousvision/occupancy_networks)
# Copyright 2019 Lars Mescheder, Michael Oechsle, Michael Niemeyer,
#                Andreas Geiger, Sebastian Nowozin
# This source code is licensed under the MIT license found in the
# 3rd-party-licenses.txt file in the root directory of this source tree.

import torch.nn as nn
from torchvision import models

from im2mesh.common import normalize_imagenet

class Resnet18(nn.Module):
    r''' ResNet-18 encoder network for image input.
    Args:
        c_dim (int): output dimension of the latent embedding
        normalize (bool): whether the input images should be normalized
        use_linear (bool): whether a final linear layer should be used
    '''

    def __init__(self, c_dim, normalize=True, use_linear=True, input_channels=3,
                 pretrained=True):
        super().__init__()
        self.normalize = normalize
        self.use_linear = use_linear
        self.features = models.resnet18(pretrained=pretrained)
        # use variable inputs, input_channels given by config.yaml
        # this messes with pretraining and initialization
        if input_channels != 3:
            self.features = models.resnet18(pretrained=pretrained)
            self.features.conv1 = nn.Conv2d(input_channels, 64,
                                            kernel_size=7, stride=2, padding=3,
                                            bias=False)
        # this creates independence of input size
        self.features.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.features.fc = nn.Sequential()
        if use_linear:
            self.fc = nn.Linear(512, c_dim)
        elif c_dim == 512:
            self.fc = nn.Sequential()
        else:
            raise ValueError('c_dim must be 512 if use_linear is False')

    def forward(self, x):
        if self.normalize:
            x = normalize_imagenet(x)
        net = self.features(x)
        out = self.fc(net)
        return out