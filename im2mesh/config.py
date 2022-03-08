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
import yaml
from torchvision import transforms

from im2mesh import data
from im2mesh import onet_sdf, onet

method_dict = {
    'onet': onet,
    'onet_sdf': onet_sdf,
}


# General config
def load_config(path, default_path=None):
    ''' Loads config file.

    Args:  
        path (str): path to config file
        default_path (bool): whether to use default path
    '''
    # Load configuration from file itself
    with open(path, 'r') as f:
        cfg_special = yaml.safe_load(f)

    # Check if we should inherit from a config
    inherit_from = cfg_special.get('inherit_from')

    # If yes, load this config first as default
    # If no, use the default_path
    if inherit_from is not None:
        cfg = load_config(inherit_from, default_path)
    elif default_path is not None:
        with open(default_path, 'r') as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = dict()

    # Include main configuration
    update_recursive(cfg, cfg_special)

    return cfg


def update_recursive(dict1, dict2):
    ''' Update two config dictionaries recursively.

    Args:
        dict1 (dict): first dictionary to be updated
        dict2 (dict): second dictionary which entries should be used

    '''
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v


# Models
def get_model(cfg, device=None, dataset=None):
    ''' Returns the model instance.

    Args:
        cfg (dict): config dictionary
        device (device): pytorch device
        dataset (dataset): dataset
    '''
    method = cfg['method']
    print(method)
    model = method_dict[method].config.get_model(
        cfg, device=device, dataset=dataset)
    return model


# Trainer
def get_trainer(model, optimizer, cfg, device):
    ''' Returns a trainer instance.

    Args:
        model (nn.Module): the model which is used
        optimizer (optimizer): pytorch optimizer
        cfg (dict): config dictionary
        device (device): pytorch device
    '''
    method = cfg['method']
    trainer = method_dict[method].config.get_trainer(
        model, optimizer, cfg, device)
    return trainer


# Generator for final mesh extraction
def get_generator(model, cfg, device):
    ''' Returns a generator instance.

    Args:
        model (nn.Module): the model which is used
        cfg (dict): config dictionary
        device (device): pytorch device
    '''
    method = cfg['method']
    generator = method_dict[method].config.get_generator(model, cfg, device)
    return generator


# Datasets
def get_dataset(mode, cfg, return_idx=False, return_category=False):
    ''' Returns the dataset.

    Args:
        model (nn.Module): the model which is used
        cfg (dict): config dictionary
        return_idx (bool): whether to include an ID field
    '''
    method = cfg['method']
    dataset_type = cfg['data']['dataset']
    dataset_folder = cfg['data']['path']
    categories = cfg['data']['classes']

    # Get split
    splits = {
        'train': cfg['data']['train_split'],
        'val': cfg['data']['val_split'],
        'test': cfg['data']['test_split'],
    }

    split = splits[mode]

    # Create dataset
    if dataset_type == 'Shapes3D':
        # Dataset fields
        # Method specific fields (usually correspond to output)
        fields = method_dict[method].config.get_data_fields(mode, cfg)
        # Input fields
        inputs_field = get_inputs_field(mode, cfg)
        if inputs_field is not None:
            fields['inputs'] = inputs_field

        if return_idx:
            fields['idx'] = data.IndexField()

        if return_category:
            fields['category'] = data.CategoryField()

        points_transform = data.SubsamplePoints(cfg['data']['points_subsample'], mode)
        dataset = data.Shapes3dDataset(
            dataset_folder, fields,
            split=split,
            categories=categories,
            transform=points_transform,
        )
    elif dataset_type == 'Shapes3DH5':
        # Dataset fields
        # Method specific fields (usually correspond to output)
        fields = method_dict[method].config.get_data_fieldsh5(mode, cfg)
        # Input fields
        inputs_field = get_inputs_fieldh5(mode, cfg)
        if inputs_field is not None:
            fields['inputs'] = inputs_field

        if return_idx:
            fields['idx'] = data.IndexFieldH5()

        if return_category:
            fields['category'] = data.CategoryFieldH5()

        points_transform = data.SubsamplePoints(cfg['data']['points_subsample'], mode)
        dataset = data.Shapes3dDatasetH5(
            dataset_folder, fields,
            split=split,
            categories=categories,
            transform=points_transform,
        )
    elif dataset_type == 'Parts3D':
        # Dataset fields
        # Method specific fields (usually correspond to output)
        fields = method_dict[method].config.get_data_fields(mode, cfg)
        # Input fields
        inputs_field = get_inputs_field(mode, cfg)
        if inputs_field is not None:
            fields['inputs'] = inputs_field

        if return_idx:
            fields['idx'] = data.IndexField()

        if return_category:
            fields['category'] = data.CategoryFieldH5()

        crop_transform = data.CropPatchAndPart(cfg, mode)
        points_transform = data.SubsamplePoints(cfg['data']['points_subsample'], mode)
        transform = transforms.Compose([
            crop_transform, points_transform
        ])

        dataset = data.Parts3dDataset(
            dataset_folder, fields,
            cfg['data']['img_size'],
            cfg['data']['parts_size'],
            cfg['data']['parts_stride'],
            mode,
            split=split,
            categories=categories,
            transform=transform,
        )

    elif dataset_type == 'Parts3DH5':
        # Dataset fields
        # Method specific fields (usually correspond to output)
        fields = method_dict[method].config.get_data_fieldsh5(mode, cfg)
        # Input fields
        inputs_field = get_inputs_fieldh5(mode, cfg)
        if inputs_field is not None:
            fields['inputs'] = inputs_field

        if return_idx:
            fields['idx'] = data.IndexFieldH5()

        if return_category:
            fields['category'] = data.CategoryFieldH5()

        crop_transform = data.CropPatchAndPart(cfg, mode)
        # subsample_transform
        points_transform = data.SubsamplePoints(cfg['data']['points_subsample'], mode)
        transform = transforms.Compose([
            crop_transform, points_transform
        ])

        dataset = data.Parts3dDatasetH5(
            dataset_folder, fields,
            cfg['data']['img_size'],
            cfg['data']['parts_size'],
            cfg['data']['parts_stride'],
            mode,
            split=split,
            categories=categories,
            transform=transform,
        )
    elif dataset_type == 'kitti':
        dataset = data.KittiDataset(
            dataset_folder, img_size=cfg['data']['img_size'],
            return_idx=return_idx
        )
    elif dataset_type == 'online_products':
        dataset = data.OnlineProductDataset(
            dataset_folder, img_size=cfg['data']['img_size'],
            classes=cfg['data']['classes'],
            max_number_imgs=cfg['generation']['max_number_imgs'],
            return_idx=return_idx, return_category=return_category
        )
    elif dataset_type == 'images':
        dataset = data.ImageDataset(
            dataset_folder, img_size=cfg['data']['img_size'],
            return_idx=return_idx,
        )
    else:
        raise ValueError('Invalid dataset "%s"' % cfg['data']['dataset'])

    return dataset


def get_inputs_field(mode, cfg):
    ''' Returns the inputs fields.

    Args:
        mode (str): the mode which is used
        cfg (dict): config dictionary
    '''
    input_type = cfg['data']['input_type']

    if input_type is None:
        inputs_field = None
    elif input_type == 'img':
        if mode == 'train' and cfg['data']['img_augment']:
            resize_op = transforms.RandomResizedCrop(
                cfg['data']['img_size'], (0.75, 1.), (1., 1.))
        else:
            resize_op = transforms.Resize((cfg['data']['img_size']))

        transform = transforms.Compose([
            resize_op, transforms.ToTensor(),
        ])

        with_camera = cfg['data']['img_with_camera']

        if mode == 'train':
            random_view = True
        else:
            random_view = False

        inputs_field = data.ImagesField(
            cfg['data']['img_folder'], transform,
            with_camera=with_camera, random_view=random_view
        )
    elif input_type == 'rgbd':
        ## see input_type == 'depth', for possible improvement of transformation
        transform = transforms.Compose([transforms.ToTensor()])
        inputs_field = data.RGBDField(cfg['data']['img_folder'],
                                      transform=transform,
                                      with_transforms=cfg['data']['img_augment']
        )
    elif input_type == 'rgbad':
        ## see input_type == 'depth', for possible improvement of transformation
        transform = transforms.Compose([transforms.ToTensor()])
        inputs_field = data.RGBADField(cfg['data']['img_folder'],
                                      transform=transform,
                                      with_transforms=cfg['data']['img_augment']
        )
    elif input_type == 'depth':

        # depth should have NN interpolation, default is 2:BILINEAR, use 0:NEAREST
        # >>> t.Resize((224,224), interpolation=PIL.Image.NEAREST)
        # Resize(size=(224, 224), interpolation=PIL.Image.NEAREST)
        # >>> t.Resize((224,224), interpolation=0)
        # Resize(size=(224, 224), interpolation=PIL.Image.NEAREST)
        resize_op = transforms.Resize((cfg['data']['img_size']), interpolation=0)

        transform = transforms.Compose([
            resize_op, transforms.ToTensor(),
        ])
        inputs_field = data.DepthField(cfg['data']['img_folder'], transform,
                                       with_transforms=cfg['data']['img_augment']
        )
    elif input_type == 'pointcloud':
        transform = transforms.Compose([
            data.SubsamplePointcloud(cfg['data']['pointcloud_n']),
            data.PointcloudNoise(cfg['data']['pointcloud_noise'])
        ])
        with_transforms = cfg['data']['with_transforms']
        inputs_field = data.PointCloudField(
            cfg['data']['pointcloud_file'], transform,
            with_transforms=with_transforms
        )
    elif input_type == 'voxels':
        inputs_field = data.VoxelsField(
            cfg['data']['voxels_file']
        )
    elif input_type == 'idx':
        inputs_field = data.IndexField()
    else:
        raise ValueError(
            'Invalid input type (%s)' % input_type)
    return inputs_field

def get_inputs_fieldh5(mode, cfg):
    ''' Returns the inputs fields.

    Args:
        mode (str): the mode which is used
        cfg (dict): config dictionary
    '''
    input_type = cfg['data']['input_type']

    if input_type is None:
        inputs_field = None
    elif input_type == 'depth':
        resize_op = transforms.Resize((cfg['data']['img_size']), interpolation=0)

        transform = transforms.Compose([
            resize_op, transforms.ToTensor(),
        ])
        inputs_field = data.DepthFieldH5(cfg['data']['img_folder'], transform,
                                       with_transforms=cfg['data']['img_augment']
        )
    else:
        raise ValueError(
            'Invalid input type (%s)' % input_type)
    return inputs_field
