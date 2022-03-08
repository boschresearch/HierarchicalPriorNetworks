# This file contains the functions to load different data modalities
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

import glob
import os
import random

import numpy as np
import torch
import trimesh
from PIL import Image

from im2mesh.data.core import Field
from im2mesh.utils import binvox_rw


class IndexField(Field):
    ''' Basic index field.'''
    def load(self, model_path, idx, category):
        ''' Loads the index field.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        return idx

    def check_complete(self, files):
        ''' Check if field is complete.

        Args:
            files: files
        '''
        return True


class CategoryField(Field):
    ''' Basic category field.'''
    def load(self, model_path, idx, category):
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


class RGBDField(Field):
    ''' RGBD Field.

    It is the field used for loading RGB and depth images.

    Args:
        folder_name (str): folder name
        transform (list): list of transformations applied to loaded depth images
        extension (str): image extension
    '''
    def __init__(self, folder_name, transform=None, with_transforms=False, extension='npz'):
        self.folder_name = folder_name
        self.transform = transform
        self.extension = extension
        self.with_transforms = with_transforms

    def load(self, model_path, idx, category):
        ''' Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        folder = os.path.join(model_path, self.folder_name)
        filename = glob.glob(os.path.join(folder, '*.%s' % self.extension))[0]

        npzfile = np.load(filename)
        if "depth_rescaled" in npzfile.files:
            depth = npzfile["depth_rescaled"]
        else:
            depth = npzfile['arr_0']

        # offset augmentation triggered by self.with_transforms
        if self.with_transforms and (np.max(depth)- np.min(depth)) > 0:
            offset = np.random.rand()
            # offset = i/10
            if offset <= 0.6:
                # rescale to 01
                depth = (depth - np.min(depth))/(np.max(depth)- np.min(depth))
                # rescale to [offset, 1]
                divisor = 1/(1-offset)
                depth /= divisor
                depth += offset

        # load image
        # as instance: png # cs instance: png # shapenet: jpg
        folder = os.path.join(model_path, "im")
        files = glob.glob(os.path.join(folder, '*.%s' % "png"))
        if len(files) == 0:
            files = glob.glob(os.path.join(folder, '*.%s' % 'jpg'))
        filename_rgb = files[0]
        rgb_image = Image.open(filename_rgb).convert('RGB')

        # image to tensor and resize in self.transform
        if self.transform is not None:
            rgb_image = self.transform(rgb_image)

        depth = depth[:, :, 2]
        depth = np.expand_dims(depth, 0)
        depth = torch.tensor(depth).float()
        image = torch.cat([rgb_image, depth], dim=0)

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


class RGBADField(Field):
    ''' RGBD Field.

    It is the field used for loading RGB, depth and alpha images.

    Args:
        folder_name (str): folder name
        transform (list): list of transformations applied to loaded depth images
        extension (str): image extension
    '''
    def __init__(self, folder_name, transform=None, with_transforms=False, extension='npz'):
        self.folder_name = folder_name
        self.transform = transform
        self.extension = extension
        self.with_transforms = with_transforms

    def load(self, model_path, idx, category):
        ''' Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        ## load depth
        folder = os.path.join(model_path, self.folder_name)
        filename = glob.glob(os.path.join(folder, '*.%s' % self.extension))[0]

        npzfile = np.load(filename)
        if "depth_rescaled" in npzfile.files:
            depth = npzfile["depth_rescaled"]
        else:
            depth = npzfile['arr_0']

        # offset augmentation triggered by self.with_transforms
        if self.with_transforms and (np.max(depth)- np.min(depth)) > 0:
            offset = np.random.rand()
            # offset = i/10
            if offset <= 0.6:
                # rescale to 01
                depth = (depth - np.min(depth))/(np.max(depth)- np.min(depth))
                # rescale to [offset, 1]
                divisor = 1/(1-offset)
                depth /= divisor
                depth += offset

        ## load image
        # as instance: png # cs instance: png # shapenet: jpg
        folder = os.path.join(model_path, "im")
        files = glob.glob(os.path.join(folder, '*.%s' % "png"))
        if len(files) == 0:
            files = glob.glob(os.path.join(folder, '*.%s' % 'jpg'))
        filename_rgb = files[0]
        rgb_image = Image.open(filename_rgb).convert('RGB')
        # image to tensor and resize in self.transform
        if self.transform is not None:
            rgb_image = self.transform(rgb_image)

        ## load alpha
        folder = os.path.join(model_path, "im_alpha")
        files = glob.glob(os.path.join(folder, '*.%s' % "png"))
        if len(files) == 0:
            files = glob.glob(os.path.join(folder, '*.%s' % 'jpg'))
        filename_rgb = files[0]
        alpha_image = Image.open(filename_rgb).convert('RGB')
        # image to tensor and resize in self.transform
        if self.transform is not None:
            alpha_image = self.transform(alpha_image)

        depth = depth[:, :, 2]
        depth = np.expand_dims(depth, 0)
        # depth = np.rollaxis(depth, 2) # 224x224x3 to 3x224x224
        depth = torch.tensor(depth).float()
        image = torch.cat([rgb_image, alpha_image, depth], dim=0)

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


class DepthField(Field):
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

    def load(self, model_path, idx, category):
        ''' Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        folder = os.path.join(model_path, self.folder_name)
        filename = glob.glob(os.path.join(folder, '*.%s' % self.extension))[0]

        with np.load(filename) as npzfile:
            if "depth_rescaled" in npzfile.files:
                depth = npzfile["depth_rescaled"]
            else:
                depth  = npzfile['arr_0']
        depth = depth[:, :, 2]

        # convert to pilimage in order to use resize transform
        depth = depth.astype(np.float32)
        pilimage = Image.fromarray(depth)
        # this is done via torchvision transforms, see im2mesh/config.py
        # pilimage = pilimage.resize((224,224), resample=Image.NEAREST)
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


class ImagesField(Field):
    ''' Image Field.

    It is the field used for loading images.

    Args:
        folder_name (str): folder name
        transform (list): list of transformations applied to loaded images
        extension (str): image extension
        random_view (bool): whether a random view should be used
        with_camera (bool): whether camera data should be provided
    '''
    def __init__(self, folder_name, transform=None,
                 extension='png', random_view=True, with_camera=False):
        self.folder_name = folder_name
        self.transform = transform
        self.extension = extension
        self.random_view = random_view
        self.with_camera = with_camera

    def load(self, model_path, idx, category):
        ''' Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        folder = os.path.join(model_path, self.folder_name)

        files = glob.glob(os.path.join(folder, '*.%s' % self.extension))
        if len(files) == 0:
            files = glob.glob(os.path.join(folder, '*.%s' % 'jpg'))

        if self.random_view:
            idx_img = random.randint(0, len(files)-1)
        else:
            idx_img = 0
        filename = files[idx_img]

        image = Image.open(filename).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        data = {
            None: image
        }

        if self.with_camera:
            camera_file = os.path.join(folder, 'cameras.npz')
            camera_dict = np.load(camera_file)
            Rt = camera_dict['world_mat_%d' % idx_img].astype(np.float32)
            K = camera_dict['camera_mat_%d' % idx_img].astype(np.float32)
            data['world_mat'] = Rt
            data['camera_mat'] = K

        return data

    def check_complete(self, files):
        ''' Check if field is complete.

        Args:
            files: files
        '''
        complete = (self.folder_name in files)
        # TODO: check camera
        return complete


# 3D Fields
class PointsField(Field):
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

    def load(self, model_path, idx, category):
        ''' Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        file_path = os.path.join(model_path, self.file_name)

        with np.load(file_path) as points_dict:
            points = points_dict['points'].astype(np.float32)
            occupancies = points_dict['occupancies']
            sdf = points_dict['sdf'].astype(np.float32)

        if self.unpackbits:
            occupancies = np.unpackbits(occupancies)[:points.shape[0]]
        occupancies = occupancies.astype(np.float32)

        data = {
            None: points,
            'occ': occupancies,
            'sdf': sdf,
            'valid':np.ones((points.shape[0]), dtype=np.bool)
        }

        if self.with_transforms:
            data['loc'] = points_dict['loc'].astype(np.float32)
            data['scale'] = points_dict['scale'].astype(np.float32)

        if self.transform is not None:
            data = self.transform(data)

        return data


class VoxelsField(Field):
    ''' Voxel field class.

    It provides the class used for voxel-based data.

    Args:
        file_name (str): file name
        transform (list): list of transformations applied to data points
    '''
    def __init__(self, file_name, transform=None):
        self.file_name = file_name
        self.transform = transform

    def load(self, model_path, idx, category):
        ''' Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        file_path = os.path.join(model_path, self.file_name)

        with open(file_path, 'rb') as f:
            voxels = binvox_rw.read_as_3d_array(f)
        voxels = voxels.data.astype(np.float32)

        if self.transform is not None:
            voxels = self.transform(voxels)

        return voxels

    def check_complete(self, files):
        ''' Check if field is complete.
        
        Args:
            files: files
        '''
        complete = (self.file_name in files)
        return complete


class PointCloudField(Field):
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

    def load(self, model_path, idx, category):
        ''' Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        file_path = os.path.join(model_path, self.file_name)

        with np.load(file_path) as pointcloud_dict:
            points = pointcloud_dict['points'].astype(np.float32)
            # normals = pointcloud_dict['normals'].astype(np.float32)
            vis_data = {}
            if 'visible' in pointcloud_dict:
                visible = pointcloud_dict['visible'].astype(np.bool)
                vis_data['visible'] = visible

        data = {
            None: points,
            # 'normals': normals,
        }
        data.update(vis_data)

        if self.with_transforms:
            data['loc'] = pointcloud_dict['loc'].astype(np.float32)
            data['scale'] = pointcloud_dict['scale'].astype(np.float32)

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


# NOTE: this will produce variable length output.
# You need to specify collate_fn to make it work with a data laoder
class MeshField(Field):
    ''' Mesh field.

    It provides the field used for mesh data. Note that, depending on the
    dataset, it produces variable length output, so that you need to specify
    collate_fn to make it work with a data loader.

    Args:
        file_name (str): file name
        transform (list): list of transforms applied to data points
    '''
    def __init__(self, file_name, transform=None):
        self.file_name = file_name
        self.transform = transform

    def load(self, model_path, idx, category):
        ''' Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        file_path = os.path.join(model_path, self.file_name)

        mesh = trimesh.load(file_path, process=False)
        if self.transform is not None:
            mesh = self.transform(mesh)

        data = {
            'verts': mesh.vertices,
            'faces': mesh.faces,
        }

        return data

    def check_complete(self, files):
        ''' Check if field is complete.
        
        Args:
            files: files
        '''
        complete = (self.file_name in files)
        return complete
