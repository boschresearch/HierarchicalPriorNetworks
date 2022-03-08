# This file contains the dataset classes reading from hdf5 files
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

import logging
import os

import h5py
import yaml
from torch.utils import data

import im2mesh.utils.parts as part_utils

logger = logging.getLogger(__name__)


# Fields
class Field(object):
    ''' Data fields class.
    '''

    def load(self, data_path, idx, category):
        ''' Loads a data point.

        Args:
            data_path (str): path to data file
            idx (int): index of data point
            category (int): index of category
        '''
        raise NotImplementedError

    def load(self, hdf5_file, model_id, idx, category):
        ''' Loads a data point.

        Args:
            data_path (str): path to data file
            idx (int): index of data point
            category (int): index of category
        '''
        raise NotImplementedError

    def check_complete(self, files):
        ''' Checks if set is complete.

        Args:
            files: files
        '''
        raise NotImplementedError


class Parts3dDatasetH5(data.Dataset):
    '''
    3D Shapes dataset class, that operates on partial shapes.
    It therefore splits the full shape (image and 3D shape)
    into parts, using a sliding window.
    '''

    def __init__(self, dataset_folder, fields, img_size, part_size, stride, mode,
                 split=None, categories=None, no_except=False, transform=None):
        ''' Initialization of the the 3D shape dataset.

        Args:
            dataset_folder (str): dataset folder
            fields (dict): dictionary of fields
            split (str): which split is used
            categories (list): list of categories to use
            no_except (bool): no exception
            transform (callable): transformation applied to data points
        '''
        print(f"Working on parts, Yeehaa! {split}")
        # Attributes
        self.dataset_folder = dataset_folder
        self.fields = fields
        self.no_except = no_except
        self.transform = transform
        self.split = split
        self.mode = mode
        self.current_hdf5_file = None
        self.current_hdf5_name = None

        # If categories is None, use all subfolders
        if categories is None:
            categories = os.listdir(dataset_folder)
            categories = [c for c in categories
                          if os.path.isdir(os.path.join(dataset_folder, c))]

        cube_indices, img_indices = part_utils.get_part_indices(img_size, part_size, stride)
        self.img_indices = img_indices
        self.cube_indices = cube_indices

        # Get all models
        self.models = []
        for c in categories: # category in our case is shapenet_full
            subpath = os.path.join(dataset_folder, c)
            if not os.path.isdir(subpath):
                logger.warning('Category %s does not exist in dataset.' % c)

            split_file = os.path.join(subpath, split + '.lst')
            with open(split_file, 'r') as f:
                models_c = f.read().splitlines()

            self.models += [
                {'category': c, 'model': m.split(' ')[0], 'hdf5': m.split(' ')[1]}
                for m in models_c
            ]
        print(f"#models {len(self.models)} * #parts {len(self.img_indices)}")

    def __len__(self):
        ''' Returns the length of the dataset.
        '''
        length_with_all_parts = len(self.models) * len(self.img_indices)
        return length_with_all_parts

    def __getitem__(self, idx):
        ''' Returns an item of the dataset.

        Args:
            idx (int): ID of data point
        '''
        model_idx = idx // len(self.img_indices)
        category = self.models[model_idx]['category']
        model = self.models[model_idx]['model']
        hdf5_name = self.models[model_idx]['hdf5']

        # when parts with modulo
        _, img_part_idx = self.get_img_part_idx(idx)
        cube_part_idx = self.get_cube_part_idx(idx)

        # TODO: performance - on avg, how many elements do we read before we switch the file?
        if self.current_hdf5_name != hdf5_name:
            # print(f"\tLoading new .h5 file: {self.current_hdf5_name} - {hdf5_name}.")
            # print(f"\tLoading new .h5 file.") # this happens all the time
            if self.current_hdf5_file is not None:
                self.current_hdf5_file.close()
            self.current_hdf5_file = h5py.File(os.path.join(self.dataset_folder, category, hdf5_name), 'r')
            self.current_hdf5_name = hdf5_name

        # possible that the index format does not work well with batching / tensors...
        data = {'img_patch_idx': img_part_idx, 'cube_part_idx': cube_part_idx}
        for field_name, field in self.fields.items():
            try:
                field_data = field.load(
                    self.current_hdf5_file, model, idx, None
                )
            except Exception:
                if self.no_except:
                    logger.warn(
                        'Error occured when loading field %s of model %s'
                        % (field_name, model)
                    )
                    return None
                else:
                    print(
                        'Error occured when loading field %s of model %s'
                        % (field_name, model)
                    )
                    raise

            if isinstance(field_data, dict):
                for k, v in field_data.items():
                    if k is None:
                        data[field_name] = v
                    else:
                        data['%s.%s' % (field_name, k)] = v
            else:
                data[field_name] = field_data

        if self.transform is not None:
            data = self.transform(data)

        return data

    def get_model_dict(self, idx):
        return self.models[idx]

    def get_img_part_idx(self, idx):
        pidx = idx % len(self.img_indices)
        return pidx, self.img_indices[pidx]

    def get_cube_part_idx(self, idx):
        pidx = idx % len(self.cube_indices)
        return self.cube_indices[pidx]

    def test_model_complete(self, category, model):
        ''' Tests if model is complete.

        Args:
            model (str): modelname
        '''
        model_path = os.path.join(self.dataset_folder, category, model)
        files = os.listdir(model_path)
        for field_name, field in self.fields.items():
            if not field.check_complete(files):
                logger.warn('Field "%s" is incomplete: %s'
                            % (field_name, model_path))
                return False

        return True


class Shapes3dDatasetH5(data.Dataset):
    ''' 3D Shapes dataset class.
    '''

    def __init__(self, dataset_folder, fields, split=None,
                 categories=None, no_except=False, transform=None):
        ''' Initialization of the the 3D shape dataset.

        Args:
            dataset_folder (str): dataset folder
            fields (dict): dictionary of fields
            split (str): which split is used
            categories (list): list of categories to use
            no_except (bool): no exception
            transform (callable): transformation applied to data points
        '''
        # Attributes
        self.dataset_folder = dataset_folder
        self.fields = fields
        self.no_except = no_except
        self.transform = transform
        self.current_hdf5_file = None
        self.current_hdf5_name = None

        # If categories is None, use all subfolders
        if categories is None:
            categories = os.listdir(dataset_folder)
            categories = [c for c in categories
                          if os.path.isdir(os.path.join(dataset_folder, c))]

        # Read metadata file
        metadata_file = os.path.join(dataset_folder, 'metadata.yaml')

        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                self.metadata = yaml.load(f)
        else:
            self.metadata = {
                c: {'id': c, 'name': 'n/a'} for c in categories
            }
            #e.g. id: 0, name: table
            #e.g. id: 1, name: chair
            #but in my case, else

        # Set index
        for c_idx, c in enumerate(categories):
            self.metadata[c]['idx'] = c_idx

        # Get all models
        self.models = []
        for c_idx, c in enumerate(categories):
            subpath = os.path.join(dataset_folder, c)
            if not os.path.isdir(subpath):
                logger.warning('Category %s does not exist in dataset.' % c)

            split_file = os.path.join(subpath, split + '.lst')
            with open(split_file, 'r') as f:
                models_c = f.read().splitlines()

            self.models += [
                {'category': c, 'model': m.split(' ')[0], 'hdf5': m.split(' ')[1]}
                for m in models_c
            ]

    def __len__(self):
        ''' Returns the length of the dataset.
        '''
        return len(self.models)

    def __getitem__(self, idx):
        ''' Returns an item of the dataset.

        Args:
            idx (int): ID of data point
        '''
        category = self.models[idx]['category']
        model = self.models[idx]['model']
        hdf5_name = self.models[idx]['hdf5']
        c_idx = self.metadata[category]['idx']

        if self.current_hdf5_name != hdf5_name:
            if self.current_hdf5_file is not None:
                self.current_hdf5_file.close()
            # TODO: performance - on avg, how many elements do we read before we switch the file?
            # print(f"\tLoading new .h5 file.")
            self.current_hdf5_file = h5py.File(os.path.join(self.dataset_folder, category, hdf5_name), 'r')
            self.current_hdf5_name = hdf5_name

        data = {}
        for field_name, field in self.fields.items():
            try:
                field_data = field.load(self.current_hdf5_file, model, idx, c_idx)
            except Exception:
                if self.no_except:
                    logger.warn(
                        'Error occured when loading field %s of model %s'
                        % (field_name, model)
                    )
                    return None
                else:
                    raise

            if isinstance(field_data, dict):
                for k, v in field_data.items():
                    if k is None:
                        data[field_name] = v
                    else:
                        data['%s.%s' % (field_name, k)] = v
            else:
                data[field_name] = field_data

        if self.transform is not None:
            data = self.transform(data)

        return data

    def get_model_dict(self, idx):
        return self.models[idx]

    def test_model_complete(self, category, model):
        ''' Tests if model is complete.

        Args:
            model (str): modelname
        '''
        model_path = os.path.join(self.dataset_folder, category, model)
        files = os.listdir(model_path)
        for field_name, field in self.fields.items():
            if not field.check_complete(files):
                logger.warn('Field "%s" is incomplete: %s'
                            % (field_name, model_path))
                return False

        return True
