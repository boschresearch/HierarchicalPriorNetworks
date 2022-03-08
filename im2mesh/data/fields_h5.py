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

import numpy as np
from PIL import Image

from im2mesh.data.core_h5 import Field


class IndexFieldH5(Field):
    ''' Basic index field.'''
    def load(self, hdf5_file, model_path, idx, category):
        ''' Loads the index field.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        return idx

class CategoryFieldH5(Field):
    ''' Basic category field.'''
    def load(self, hdf5_file, model_path, idx, category):
        ''' Loads the category field.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        return category

    def check_complete(self, files):
        ''' Check if field is complete.

        Args:
            files: files
        '''
        return True

class DepthFieldH5(Field):
    ''' Depth Field.

    It is the field used for loading depth images.

    Args:
        folder_name (str): folder name
        transform (list): list of transformations applied to loaded depth images
        extension (str): image extension
    '''
    def __init__(self, folder_name, transform, with_transforms=False, extension='npz'):
        self.folder_name = folder_name
        self.transform = transform
        self.extension = extension
        self.with_transforms = with_transforms

    def load(self, hdf5_file, model_id, idx, category):
        ''' Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        g_depth = hdf5_file.get("depth")
        depth = g_depth.get(model_id)
        depth = np.asarray(depth)[:, :, 2]

        depth = depth.astype(np.float32)
        pilimage = Image.fromarray(depth)
        # this is done via torchvision transforms
        # pilimage = pilimage.resize((224,224), resample=Image.Nearest)

        if self.transform is not None:
            image = self.transform(pilimage)

        data = {
            None: image
        }
        return data

    def check_complete(self, files):
        ''' Check if field is complete.

        Args:
            files: files
        '''
        complete = (self.folder_name in files)
        return complete

# 3D Fields
class PointsFieldH5(Field):
    ''' Point Field.

    It provides the field to load point data. This is used for the points
    randomly sampled in the bounding volume of the 3D shape.

    Args:
        file_name (str): file name
        transform (list): list of transformations which will be applied to the
            points tensor
        with_transforms (bool): whether scaling and rotation data should be
            provided

    '''
    def __init__(self, file_name, transform=None, with_transforms=False, unpackbits=False):
        self.file_name = file_name
        self.transform = transform
        self.with_transforms = with_transforms
        self.unpackbits = unpackbits

    def load(self, hdf5_file, model_id, idx, category, points_part=None):
        ''' Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        g_points = hdf5_file.get("points")
        g_occupancy = hdf5_file.get("occupancy")
        g_sdf = hdf5_file.get("sdf")
        
        points = np.asarray(g_points.get(model_id), dtype=np.float32)
        occupancies = np.asarray(g_occupancy.get(model_id), dtype=np.float32)
        sdf = np.asarray(g_sdf.get(model_id), dtype=np.float32)

        data = {
            None: points,
            'occ': occupancies,
            'sdf': sdf,
            'valid':np.ones((points.shape[0]), dtype=np.bool)
        }

        if self.with_transforms:
            data['loc'] = hdf5_file.get('loc').astype(np.float32)
            data['scale'] = hdf5_file.get('scale').astype(np.float32)

        if self.transform is not None:
            data = self.transform(data)

        return data

class PointCloudFieldH5(Field):
    ''' Point cloud field.

    It provides the field used for point cloud data. These are the points
    randomly sampled on the mesh.

    Args:
        file_name (str): file name
        transform (list): list of transformations applied to data points
        with_transforms (bool): whether scaling and rotation dat should be
            provided
    '''
    def __init__(self, file_name, transform=None, with_transforms=False):
        self.file_name = file_name
        self.transform = transform
        self.with_transforms = with_transforms

    def load(self, hdf5_file, model_id, idx, category):
        ''' Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        g_pointclouds = hdf5_file.get("pointclouds")
        
        points = np.asarray(g_pointclouds.get(model_id), dtype=np.float32)

        data = {
            None: points
        }

        if self.transform is not None:
            data = self.transform(data)

        return data

    def check_complete(self, files):
        ''' Check if field is complete.
        
        Args:
            files: files
        '''
        complete = (self.file_name in files)
        return complete
