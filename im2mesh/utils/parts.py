# This file contains the functionality for local img and 3D patches
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
import os
import shutil
from typing import Dict, Tuple, List, Union

import numpy as np
import pandas as pd
import torch

from im2mesh import config
from im2mesh.checkpoints import CheckpointIO
from im2mesh.utils.visualize import visualize_data


def make_gaussian(shape, sigma=1):
    """ create a two dimensional gaussian weighting mask"""
    x = np.arange(0, shape, 1, float)
    g = np.exp(-4*np.log(2) * (x-(shape//2))**2/(sigma**2))
    g2d = np.outer(g, g)
    return g2d


def plot_img_from_vg_z(value_grid, image):
    import matplotlib
    matplotlib.use('Qt4Agg')
    import matplotlib.pyplot as plt
    ind = np.unravel_index(np.argmax(value_grid, axis=None), value_grid.shape)
    zmax = ind[-1]
    img_z = value_grid[:, :, zmax:zmax+1] # -> gives Himg=x, Wimg=z
    img_z = img_z.squeeze()

    plt.subplot(1,2,1)
    plt.title(f"depth channel {zmax}: {img_z.min(): 2.2f} {img_z.max(): 2.2f}")
    plt.imshow(img_z)
    plt.subplot(1,2,2)
    plt.title("input")
    img = image.copy()
    plt.imshow(img)
    plt.show()


def plot_img_from_vg(value_grid, image, slice_h=64, slice_v=75, calibrate=True): # black slice_v=75 slice_v=10
    """ show logits as image by slicing through value grid and
           3
          /
         /
        x------->1
        |
        |
        v 2
    """
    import matplotlib
    matplotlib.use('Qt4Agg')
    import matplotlib.pyplot as plt
    img_horizontal = value_grid[:, slice_h:slice_h+1, :] # -> gives Himg=x, Wimg=z
    img_horizontal = img_horizontal.squeeze()
    img_vertical = value_grid[slice_v:slice_v+1, :, :] # -> gives Himg=y, Wimg=z
    img_vertical = img_vertical.squeeze()
    plt.subplot(1,3,1)
    plt.title(f"horizontal {img_horizontal.min(): 2.2f} {img_horizontal.max(): 2.2f}")
    if calibrate:
        img_horizontal[0,0] = -20 # calibrate colormap
        img_horizontal[0,1] = 10
    img_horizontal = np.clip(img_horizontal, -20, 10)
    plt.imshow(img_horizontal)
    plt.subplot(1,3,2)
    plt.title(f"vertical {img_vertical.min(): 2.2f} {img_vertical.max(): 2.2f}")
    if calibrate:
        img_vertical[0,0] = -20
        img_vertical[0,1] = 10
    img_vertical = np.clip(img_vertical, -20, 10)
    plt.imshow(img_vertical)
    plt.subplot(1,3,3)
    plt.title("input")
    # draw cut into image
    height, width = image.shape
    vg_size = value_grid.shape[0]
    img_h_idx = int(slice_h / vg_size * height)
    img_v_idx = int(slice_v / vg_size * width)
    print(f"{img_h_idx} {img_v_idx}")
    img = image.copy()
    img[img_h_idx:img_h_idx+1, :] = image.max()
    img[:, img_v_idx:img_v_idx+1] = image.max()
    plt.imshow(img)
    plt.show()


def collect_cube_mesh(mesh, cube_min, cube_max):
    if mesh.is_empty:
        return
    # transform: [-0.5, 0.5]^3 to index size and position
    vertices = mesh.vertices
    cube_size = cube_max - cube_min
    print(f"min{np.min(vertices, axis=0)} max {np.max(vertices, axis=0)}")
    vertices = ((vertices + np.asarray([0.55, 0.55, 0.55])) * cube_size) + cube_min
    # vertices = ((vertices + np.min(vertices, axis=0)) * cube_size) + cube_min
    print(f"min{np.min(vertices, axis=0)} max {np.max(vertices, axis=0)}")

    mesh.vertices = vertices
    print(mesh)

    import open3d as o3d
    mesh.export("temp_mesh.ply") # what changes here?
    omesh = o3d.read_triangle_mesh("temp_mesh.ply")
    omesh.compute_vertex_normals()
    omesh.paint_uniform_color(np.random.rand(3))
    o3d.draw_geometries([omesh])


class Settings(object):
    def __init__(self, config_file, device, part=False): # decide part based on config dataset
        """ initialize all settings """
        cfg = config.load_config(config_file, 'configs/default.yaml')
        out_dir = cfg['training']['out_dir']
        self.generation_dir = os.path.join(out_dir, cfg['generation']['generation_dir'])

        self.input_type = cfg['data']['input_type']
        self.vis_n_outputs = cfg['generation']['vis_n_outputs']
        if self.vis_n_outputs is None:
            self.vis_n_outputs = -1

        # Dataset
        self.dataset = config.get_dataset('test', cfg, return_idx=True)
        if part:
            batch_size = len(self.dataset.img_indices) # has to be 49
        else:
            batch_size = 1

        # Model
        self.model = config.get_model(cfg, device=device, dataset=self.dataset)

        checkpoint_io = CheckpointIO(out_dir, model=self.model)
        checkpoint_io.load(cfg['test']['model_file'], map_location=device)

        # Generator
        self.generator = config.get_generator(self.model, cfg, device=device)

        # Loader
        self.test_loader = torch.utils.data.DataLoader(
            self.dataset, batch_size=batch_size, num_workers=0, shuffle=False)
        # Generate
        self.model.eval()
        self.cfg = cfg

    @staticmethod
    def _assert_compatible_pair(setting1, setting2):
        # model files
        assert setting1.cfg['data']['test_split'] ==\
               setting2.cfg['data']['test_split'],\
            f"{setting1.cfg['data']['test_split']} != " \
            f"{setting2.cfg['data']['test_split']}"
        assert setting1.cfg['data']['train_split'] ==\
               setting2.cfg['data']['train_split'],\
            f"{setting1.cfg['data']['train_split']} !" \
            f"{setting2.cfg['data']['train_split']}"
        # dataset path
        assert setting1.cfg['data']['path'] ==\
               setting2.cfg['data']['path'],\
            f"{setting1.cfg['data']['path']} != " \
            f"{setting2.cfg['data']['path']}"
        # category
        assert setting1.cfg['data']['classes'][0] ==\
               setting2.cfg['data']['classes'][0],\
            f"{setting1.cfg['data']['classes'][0]} != " \
            f"{setting2.cfg['data']['classes'][0]}"
        # first model name
        assert setting1.dataset.get_model_dict(0)['model'] ==\
               setting2.dataset.get_model_dict(0)['model'],\
            f"{setting1.dataset.get_model_dict(0)['model']} != " \
            f"{setting2.dataset.get_model_dict(0)['model']}"

    @staticmethod
    def assert_compatible(setting_list: List) -> None:
        """ Make sure that the settings specify the same test data
        Args:
            setting_list: a list of two or more settings
        """
        assert len(setting_list) > 1, "Minimum 2 settings required."
        for i in range(len(setting_list) - 1):
            Settings._assert_compatible_pair(setting_list[i], setting_list[i+1])

    @staticmethod
    def assert_compatible_datapair(setting1, setting2,
                                   data1: Dict[str, torch.Tensor],
                                   data2: Dict[str, torch.Tensor]) -> Tuple[Dict, str]:
        patch_model_dict = setting1.dataset.get_model_dict(data1['idx'][0])
        patch_modelname = patch_model_dict['model']

        model_dict = setting2.dataset.get_model_dict(data2['idx'].item())
        modelname = model_dict['model']
        assert patch_modelname == modelname, f"\n" \
                                             f"\tpatch-{patch_modelname} != bl-{modelname} \n" \
                                             f"\t{setting1.cfg['data']['test_split']}\n" \
                                             f"\t{setting2.cfg['data']['test_split']}"
        return model_dict, modelname

    def save_inputs(self, in_dir, inputs, modelname, out_file_dict):
        if self.cfg['generation']['copy_input']:
            # Save inputs
            # TODO: fuse these two -> solve image format - can I specify it?
            if self.input_type in ('img', 'depth'):
                inputs_path = os.path.join(in_dir, '%s.jpg' % modelname)
                visualize_data(inputs, 'img', inputs_path)
                out_file_dict['in'] = inputs_path
            elif self.input_type == 'rgbd':
                inputs_path = os.path.join(in_dir, '%s.png' % modelname)
                visualize_data(inputs, 'rgbd', inputs_path)
                out_file_dict['in'] = inputs_path


def get_valuegrid_patch(data, setting):
    """ iterate over the batch and generate value grid
    return value_grid (np.ndarray) resolution * 2^ upsampling
    """
    # set patch constants
    PART_SIZE = setting.cfg['data']['parts_size']
    IMAGE_DIM = setting.cfg['data']['img_size']
    FACTOR = int(IMAGE_DIM / PART_SIZE)

    res0 = setting.cfg['generation']['resolution_0']
    upsample = setting.cfg['generation']['upsampling_steps']# + 1
    VALG_DIM = res0 * 2 ** upsample
    # 32 for patch64, 16 for patch32
    # print(VALG_DIM) # or set this directly to a small res

    big_val_grid = np.ones((VALG_DIM*FACTOR, VALG_DIM*FACTOR, VALG_DIM*FACTOR)) * np.NINF # some small neg number should be fine though
    big_val_count = np.zeros((VALG_DIM*FACTOR, VALG_DIM*FACTOR, VALG_DIM*FACTOR)) # count the overlaps this cell has seen

    GAUSS_COV = VALG_DIM//2
    # print(f"gauss cov: {GAUSS_COV}")
    gauss2d = make_gaussian(VALG_DIM, sigma=GAUSS_COV)
    gauss2d = np.expand_dims(gauss2d, -1)
    # assemble input as well
    input_type = setting.input_type
    if input_type == "img":
        image = np.ones((3,256,256)) # rgb
    elif input_type == "depth":
        image = np.ones((256,256)) # depth
    else:
        print(f"unknown input image type {input_type}")
        exit()
    # process one batch
    for idx, img_patch in zip(data['idx'], data['inputs']):
        img_patch_np = img_patch.cpu().numpy().squeeze()
        y0, x0, y1, x1 = setting.dataset.get_img_part_idx(idx)[1]
        if input_type == "img":
             # this works also for overlapping parts
            image[:, y0:y1, x0:x1] = img_patch_np # rgb
        elif input_type == "depth":
            # depth
            image[y0:y1, x0:x1] = img_patch_np
        if (img_patch_np >= 0).sum() == 0: # skip empty parts
            continue

        # get 3D cube coordinates to stack in continuous 3D (mesh/points)
        cube_part = setting.dataset.get_cube_part_idx(idx)
        # img indices  |   cube indices   -->   value grid content
        # --------> 2  |        ^                2
        # |            |        |                ^
        # |            |   -----|----->          |
        # v            |        |                |
        # 1            |        |                -----------> 1
        cube_min, cube_max = cube_part
        cube_min = (cube_min + 0.5) * VALG_DIM * FACTOR
        cube_max = (cube_max + 0.5) * VALG_DIM * FACTOR
        xmin = cube_min[0]
        xmax = cube_max[0]
        ymin = cube_min[1]
        ymax = cube_max[1]

        value_grid, _ = setting.generator.generate_value_grid({'inputs':img_patch.unsqueeze(0)})

        # apply gaussian weighting
        value_grid *= gauss2d

        # create view into current part of big val grid
        # print(f"{int(xmin)}:{int(xmax)}, {int(ymin)}:{int(ymax)}")
        view_big_part = big_val_grid[int(xmin):int(xmax), int(ymin):int(ymax), :]
        # update the count
        # big_val_count[int(xmin):int(xmax), int(ymin):int(ymax), :] += 1
        big_val_count[int(xmin):int(xmax), int(ymin):int(ymax), :] += gauss2d
        # get ninf mask
        mask_ninf = np.isneginf(view_big_part)
        # already available values
        view_big_part[~mask_ninf] = value_grid[~mask_ninf] + view_big_part[~mask_ninf]
        # set values of current part to the ones that are ninf in big (without mean)
        view_big_part[mask_ninf] = value_grid[mask_ninf]

    mask_big = np.isneginf(big_val_grid)
    big_val_grid[~mask_big] /= big_val_count[~mask_big]
    # replace -inf with -10, otherwise Marching Cubes creates NaN values for vertices
    big_val_grid[mask_big] = -10
    # mesh = generator.extract_mesh(big_val_grid, 0, stats_dict=stats_dict)
    return big_val_grid, torch.tensor(image)


def get_valuegrid_baseline(data, setting):
    """ iterate over the batch and generate value grid
    return value_grid (np.ndarray) resolution * 2^ upsampling
    """
    idx = data['idx'].item()
    model_dict = setting.dataset.get_model_dict(idx)
    category_id = model_dict.get('category', 'n/a')
    value_grid, _ = setting.generator.generate_value_grid(data)
    input_image = data['inputs'].squeeze(0).cpu()
    return value_grid, input_image


def rescale_patchgrid(vg_local, goal_range, threshold_logit, threshold_sdf):
    # center at threshold
    vg_local -= threshold_logit
    # rescale
    vg_local = vg_local / max(abs(vg_local.min()), abs(vg_local.max()))
    # determine which one we took
    minimum = abs(vg_local.min()) > abs(vg_local.max())
    # then take min or max from range accordingly
    if minimum:
        scale = abs(goal_range[0] - threshold_sdf)
    else:
        scale = abs(goal_range[1] - threshold_sdf)
    # then scale by min or max - thresholdsdf
    vg_local *= scale
    # then subtract thresholdsdf
    vg_local += threshold_sdf
    return vg_local


def plot_logits(bl_setting, combined, vg_global, patch_setting, vg_local, data_bl):
    import open3d as o3d
    mesh = bl_setting.generator.extract_mesh(combined, 0)
    mesh1 = bl_setting.generator.extract_mesh(vg_global, 0)
    mesh2 = patch_setting.generator.extract_mesh(vg_local, 0)
    mesh.export("temp_mesh.ply")  # save to agree in format trimesh open3d
    omesh = o3d.read_triangle_mesh("temp_mesh.ply")
    mesh1.export("temp_mesh.ply")  # save to agree in format trimesh open3d
    omesh1 = o3d.read_triangle_mesh("temp_mesh.ply")
    mesh2.export("temp_mesh.ply")  # save to agree in format trimesh open3d
    omesh2 = o3d.read_triangle_mesh("temp_mesh.ply")
    omesh.compute_vertex_normals()
    omesh1.compute_vertex_normals()
    omesh2.compute_vertex_normals()
    o3d.draw_geometries([omesh])
    o3d.draw_geometries([omesh1])
    o3d.draw_geometries([omesh2])
    # plot
    image = data_bl['inputs'].squeeze().cpu().numpy()
    print(image.shape)
    threshold = np.log(0.2) - np.log(1. - 0.2)
    # global
    vg_global_abs = vg_global.copy()
    vg_global_abs[vg_global_abs < threshold] = -3  # make binary
    vg_global_abs[vg_global_abs >= threshold] = 3
    plot_img_from_vg(vg_global_abs, image, calibrate=False)
    # local
    vg_local_abs = vg_local.copy()
    vg_local_abs[vg_local_abs < threshold] = -3
    vg_local_abs[vg_local_abs >= threshold] = 3
    plot_img_from_vg(vg_local_abs, image)
    # combined
    combined_abs = combined.copy()
    combined_abs[combined_abs < threshold] = -3
    combined_abs[combined_abs >= threshold] = 3
    plot_img_from_vg(combined_abs, image)
    exit()


class IOHelper(object):
    """ class that contains all output directories for generation """
    def __init__(self, patch_setting, gen_dir_apx: str):
        """ set the initial directories
        How to typing for patch setting?
        Args:
            patch_setting (object): determines the exp folder where generations are saved
            gen_dir_apx (str): appendix for merge, e.g. _hierarchical"""

        self.generation_dir = patch_setting.generation_dir + gen_dir_apx
        self.out_time_file = os.path.join(self.generation_dir, 'time_generation_full.pkl')
        self.out_time_file_class = os.path.join(self.generation_dir, 'time_generation.pkl')
        self.mesh_dir = None
        self.in_dir = None
        self.generation_vis_dir = None
        self.modelname = None
        self.category_id = None

    def create_out_dirs(self, category_id: str, category_name: str, create_vis_dir: bool=False):
        """ sets and creates the output directories per category

        Args:
            category_id: dataset name
            category_name: typically n/a for us
            create_vis_dir (bool): whether to have a separate vis dir or not
        """
        # Output folders
        mesh_dir = os.path.join(self.generation_dir, 'meshes')
        in_dir = os.path.join(self.generation_dir, 'input')
        generation_vis_dir = os.path.join(self.generation_dir, 'vis')

        if category_id != 'n/a':
            mesh_dir = os.path.join(mesh_dir, category_id)
            in_dir = os.path.join(in_dir, category_id)
            folder_name = category_id
            if category_name != 'n/a':
                folder_name = folder_name + '_' + category_name.split(',')[0]
            generation_vis_dir = os.path.join(generation_vis_dir, folder_name)
        # Create directories if necessary
        if create_vis_dir and not os.path.exists(generation_vis_dir):
            os.makedirs(generation_vis_dir)
        if not os.path.exists(mesh_dir):
            os.makedirs(mesh_dir)
        if not os.path.exists(in_dir):
            os.makedirs(in_dir)

        self.mesh_dir = mesh_dir
        self.in_dir = in_dir
        self.generation_vis_dir = generation_vis_dir

    def update_fields_from_data(self, bl_setting, it):
        model_dict = bl_setting.dataset.get_model_dict(it)
        self.modelname = model_dict['model']
        self.category_id = model_dict.get('category', 'n/a') # shapenet_hdf5
        try:
            category_name = bl_setting.dataset.metadata[self.category_id].get('name', 'n/a')
        except AttributeError:
            category_name = 'n/a'

        self.create_out_dirs(self.category_id, category_name, create_vis_dir=(bl_setting.vis_n_outputs > 0))

        # Timing dict
        time_dict = {
            'idx': it,
            'class id': self.category_id,
            'class name': category_name,
            'modelname': self.modelname,
        }

        return time_dict

    def copy_to_vis(self, c_it, setting, out_file_dict):
        if c_it < setting.vis_n_outputs:
            for k, filepath in out_file_dict.items():
                ext = os.path.splitext(filepath)[1]
                out_file = os.path.join(self.generation_vis_dir, '%02d_%s%s'
                                        % (c_it, k, ext))
                shutil.copyfile(filepath, out_file)

    def save_df(self, time_dicts):
        time_df = pd.DataFrame(time_dicts)
        time_df.set_index(['idx'], inplace=True)
        time_df.to_pickle(self.out_time_file)

        # Create pickle files with main statistics
        time_df_class = time_df.groupby(by=['class name']).mean()
        time_df_class.to_pickle(self.out_time_file_class)

        # Print results
        time_df_class.loc['mean'] = time_df_class.mean()
        print('Timings [s]:')
        print(time_df_class)


def get_part_indices(img_size, part_size, stride)\
        -> Tuple[List[Tuple[Union[np.ndarray, np.ndarray], Union[np.ndarray, np.ndarray]]],
                 List[Tuple[int, int, int, int]]]:
    """ perform a sliding window over the img for given part size and stride

    Args:
        img_size (int): image size as specified in config
        part_size (int): size of square image patch
        stride (int): step size of sliding window, typically parts_size/2
        """
    img_dim = (img_size, img_size)
    size_y = size_x = part_size
    cube_window_x = size_x / img_dim[1]
    cube_window_y = size_y / img_dim[0]

    assert img_size > part_size, f"img should be larger than part"
    assert (img_size - part_size) % stride == 0,\
        f"\nYou'll miss parts of the image. " \
        f"No problem for training, but likely for generation. \n" \
        f"Choose part size and stride s.t. (img_size - part_size) % stride == 0"

    img_indices = []
    cube_indices = []
    # ------ sliding window without padding
    for i in range(0, img_dim[0], stride):
        for j in range(0, img_dim[1], stride):
            if i + size_y > img_dim[0] or j + size_x > img_dim[1]:
                break
            img_indices.append((i, j, i + size_y, j + size_x))

            cube_idx = j / img_dim[1] - 0.5
            cube_idy = (i / img_dim[0] - 0.5) * -1
            # img min is top left - get bottom left for cube
            cube_idy -= cube_window_y
            cube_min = np.asarray([cube_idx, cube_idy, -0.51])
            # img max is bottom right - get top right for cube
            cube_max = np.asarray([cube_idx + cube_window_x, cube_idy + cube_window_y, 0.51])
            cube_indices.append((cube_min, cube_max))
    return cube_indices, img_indices
