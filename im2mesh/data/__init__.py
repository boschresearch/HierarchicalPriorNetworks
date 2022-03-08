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

from im2mesh.data.core import (
    Shapes3dDataset, collate_remove_none, worker_init_fn, Parts3dDataset
)
# h5 stuff
from im2mesh.data.core_h5 import (
    Shapes3dDatasetH5, Parts3dDatasetH5
)
from im2mesh.data.fields import (
    IndexField, CategoryField, ImagesField, DepthField, RGBDField, RGBADField,
    PointsField, VoxelsField, PointCloudField, MeshField,
)
from im2mesh.data.fields_h5 import (
    DepthFieldH5, PointsFieldH5, PointCloudFieldH5, IndexFieldH5, CategoryFieldH5,
)
from im2mesh.data.real import (
    KittiDataset, OnlineProductDataset,
    ImageDataset,
)
from im2mesh.data.transforms import (
    PointcloudNoise, SubsamplePointcloud,
    SubsamplePoints, CropPatchAndPart
)

__all__ = [
    # Core
    Shapes3dDataset,
    Parts3dDataset,
    collate_remove_none,
    worker_init_fn,
    # CoreH5
    Shapes3dDatasetH5,
    Parts3dDatasetH5,
    # Fields
    IndexField,
    CategoryField,
    ImagesField,
    DepthField,
    RGBDField,
    RGBADField,
    PointsField,
    VoxelsField,
    PointCloudField,
    MeshField,
    # FieldsH5
    IndexFieldH5,
    CategoryFieldH5,
    DepthFieldH5,
    PointsFieldH5,
    PointCloudFieldH5,
    # Transforms
    PointcloudNoise,
    SubsamplePointcloud,
    SubsamplePoints,
    CropPatchAndPart,
    # Real Data
    KittiDataset,
    OnlineProductDataset,
    ImageDataset,
]
