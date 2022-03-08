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
import argparse
import os

import open3d  # must: import before torch, used later in eval.py
import pandas as pd
import torch
import trimesh
from tqdm import tqdm

from im2mesh import config, data
from im2mesh.eval import MeshEvaluator
from im2mesh.utils.io import load_pointcloud

LOG_SCORES_REDUCED = ['fscore@1 (mesh)', 'chamfer (mesh)', 'iou (mesh)']
LOG_SCORES_ALL = [
    'fscore@1 (mesh)', 'chamfer (mesh)', 'iou (mesh)',
    'fscore@1_vis (mesh)', 'chamfer_vis (mesh)',
    'fscore@1_invis (mesh)', 'chamfer_invis (mesh)'
]
log_scores = LOG_SCORES_REDUCED

parser = argparse.ArgumentParser(
    description='Evaluate mesh algorithms.'
)
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
parser.add_argument('--eval_input', action='store_true',
                    help='Evaluate inputs instead.')

args = parser.parse_args()
cfg = config.load_config(args.config, 'configs/default.yaml')
is_cuda = (torch.cuda.is_available() and not args.no_cuda)
device = torch.device("cuda" if is_cuda else "cpu")

# Shorthands
testset = cfg['data']['classes'][0]
out_dir = cfg['training']['out_dir']
generation_dir = os.path.join(out_dir, cfg['generation']['generation_dir'])
if not args.eval_input:
    out_file = os.path.join(generation_dir, f'eval_meshes_full_{testset}.pkl')
    out_file_class = os.path.join(generation_dir, f'eval_meshes_{testset}.csv')
else:
    out_file = os.path.join(generation_dir, 'eval_input_full.pkl')
    out_file_class = os.path.join(generation_dir, 'eval_input.csv')

# Dataset
dataset_type = cfg['data']['dataset']
# TODO: replace this with a function (e.g. dataset = config.get_dataset('test', cfg))
# TODO: why is this separate dataset setup necessary??
if 'H5' in dataset_type:
    points_field = data.PointsFieldH5(
        cfg['data']['points_iou_file'],
        unpackbits=cfg['data']['points_unpackbits'],
    )
    pointcloud_field = data.PointCloudFieldH5(
        cfg['data']['pointcloud_chamfer_file']
    )
    fields = {
        'points_iou': points_field,
        'pointcloud_chamfer': pointcloud_field,
        'idx': data.IndexFieldH5(),
    }

    print('Test split: ', cfg['data']['test_split'])

    dataset_folder = cfg['data']['path']
    dataset = data.Shapes3dDatasetH5(
        dataset_folder, fields,
        cfg['data']['test_split'],
        categories=cfg['data']['classes'])
else:
    points_field = data.PointsField(
        cfg['data']['points_iou_file'],
        unpackbits=cfg['data']['points_unpackbits'],
    )
    pointcloud_field = data.PointCloudField(
        cfg['data']['pointcloud_chamfer_file']
    )
    fields = {
        'points_iou': points_field,
        'pointcloud_chamfer': pointcloud_field,
        'idx': data.IndexField(),
    }

    print('Test split: ', cfg['data']['test_split'])

    dataset_folder = cfg['data']['path']
    dataset = data.Shapes3dDataset(
        dataset_folder, fields,
        cfg['data']['test_split'],
        categories=cfg['data']['classes'])

# Evaluator
evaluator = MeshEvaluator(n_points=100000)

# Loader
test_loader = torch.utils.data.DataLoader(
    dataset, batch_size=1, num_workers=0, shuffle=False)

# Evaluate all classes
eval_dicts = []
print('Evaluating meshes...')
for it, data in enumerate(tqdm(test_loader)):
    if data is None:
        print('Invalid data.')
        continue

    # Output folders
    if not args.eval_input:
        mesh_dir = os.path.join(generation_dir, 'meshes')
        pointcloud_dir = os.path.join(generation_dir, 'pointcloud')
    else:
        mesh_dir = os.path.join(generation_dir, 'input')
        pointcloud_dir = os.path.join(generation_dir, 'input')

    # Get index etc.
    idx = data['idx'].item()

    try:
        model_dict = dataset.get_model_dict(idx)
    except AttributeError:
        model_dict = {'model': str(idx), 'category': 'n/a'}

    modelname = model_dict['model']
    category_id = model_dict['category']

    try:
        category_name = dataset.metadata[category_id].get('name', 'n/a')
    except AttributeError:
        category_name = 'n/a'

    if category_id != 'n/a':
        mesh_dir = os.path.join(mesh_dir, category_id)
        pointcloud_dir = os.path.join(pointcloud_dir, category_id)

    # Evaluate
    pointcloud_tgt = data['pointcloud_chamfer'].squeeze(0).numpy()
    normals_tgt = None # data['pointcloud_chamfer.normals'].squeeze(0).numpy()
    pointcloud_tgt_visibility = None
    if 'pointcloud_chamfer.visible' in data:
        pointcloud_tgt_visibility = data['pointcloud_chamfer.visible'].squeeze(0).numpy()
        log_scores = LOG_SCORES_ALL
    points_tgt = data['points_iou'].squeeze(0).numpy()
    occ_tgt = data['points_iou.occ'].squeeze(0).numpy()

    # Evaluating mesh and pointcloud
    # Start row and put basic informatin inside
    eval_dict = {
        'idx': idx,
        'class id': category_id,
        # 'class name': category_name,
        'class name': modelname.split('_')[0],
        'modelname': modelname,
    }
    eval_dicts.append(eval_dict)

    # Evaluate mesh
    if cfg['test']['eval_mesh']:
        mesh_file = os.path.join(mesh_dir, '%s.off' % modelname)

        if os.path.exists(mesh_file):
            mesh = trimesh.load(mesh_file, process=False)
            eval_dict_mesh = evaluator.eval_mesh(
                mesh, pointcloud_tgt, normals_tgt, points_tgt, occ_tgt,
                pointcloud_tgt_visibility)
            for k, v in eval_dict_mesh.items():
                eval_dict[k + ' (mesh)'] = v
        else:
            print('Warning: mesh does not exist: %s' % mesh_file)

    # Evaluate point cloud
    if cfg['test']['eval_pointcloud']:
        pointcloud_file = os.path.join(
            pointcloud_dir, '%s.ply' % modelname)

        if os.path.exists(pointcloud_file):
            pointcloud = load_pointcloud(pointcloud_file)
            eval_dict_pcl = evaluator.eval_pointcloud(
                pointcloud, pointcloud_tgt)
            for k, v in eval_dict_pcl.items():
                eval_dict[k + ' (pcl)'] = v
        else:
            print('Warning: pointcloud does not exist: %s'
                    % pointcloud_file)

# Create pandas dataframe and save
eval_df = pd.DataFrame(eval_dicts)
eval_df.set_index(['idx'], inplace=True)
eval_df.to_pickle(out_file)

# Create CSV file  with main statistics
eval_df_class = eval_df.groupby(by=['class name']).mean()
eval_df_class.to_csv(out_file_class)
# Print results
eval_df_class.loc['mean'] = eval_df_class.mean()
eval_df_class.loc['mean_sample'] = eval_df.mean()
print(eval_df_class[log_scores])
