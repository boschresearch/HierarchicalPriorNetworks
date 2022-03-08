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
import argparse
import os
import time
from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm

from im2mesh.utils.parts import Settings, get_valuegrid_patch, \
    plot_logits, IOHelper

ISDEBUG = False

parser = argparse.ArgumentParser(
    description='Extract meshes from occupancy process.'
)
parser.add_argument('config_patch1', type=str, help='Path to patch config file.')
parser.add_argument('config_patch2', type=str, help='Path to patch config file.')
parser.add_argument('config_patch3', type=str, help='Path to baseline config file.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')

args = parser.parse_args()

if "p32s16" in args.config_patch1:
    print("Be careful, normally HPN is stored in p32s16/hierarchical3")
    exit()


# common settings
is_cuda = (torch.cuda.is_available() and not args.no_cuda)
device = torch.device("cuda" if is_cuda else "cpu")
# Statistics
time_dicts: List[Dict] = []

# get setup
patch_setting1 = Settings(args.config_patch1, device, part=True)
patch_setting2 = Settings(args.config_patch2, device, part=True)
patch_setting3 = Settings(args.config_patch3, device, part=True)
Settings.assert_compatible([patch_setting1, patch_setting2, patch_setting3])

# generation directory
ioh = IOHelper(patch_setting1, "_hierarchical3")

# Count how many models already created
model_counter = defaultdict(int) # make this the full image model counter (batch)

for it, (data_patch1, data_patch2, data_patch3) in enumerate(
                                tqdm(zip(
                                    patch_setting1.test_loader,
                                    patch_setting2.test_loader,
                                    patch_setting3.test_loader),
                                total=len(patch_setting1.test_loader))):
    time_dict = ioh.update_fields_from_data(patch_setting1, it)
    time_dicts.append(time_dict)

    # Generate outputs
    out_file_dict = {}

    t0 = time.time()
    time_dict['mesh'] = time.time() - t0
    vg_local1, image = get_valuegrid_patch(data_patch1, patch_setting1)
    vg_local2, _ = get_valuegrid_patch(data_patch2, patch_setting2)
    vg_local3, _ = get_valuegrid_patch(data_patch3, patch_setting3)
    combined = (vg_local1 + vg_local2 + vg_local3) / 3
    padded_big_val_grid = np.ones((combined.shape[0]+2,combined.shape[1]+2,combined.shape[2]+2)) -10
    padded_big_val_grid[1:-1,1:-1,1:-1] = combined
    mesh = patch_setting1.generator.extract_mesh(padded_big_val_grid,0)
    if ISDEBUG:
        plot_logits(patch_setting1, combined, vg_local1, patch_setting2, vg_local2, data_patch1)

    mesh_out_file = os.path.join(ioh.mesh_dir, '%s.ply' % ioh.modelname)
    mesh.export(mesh_out_file)
    out_file_dict['mesh'] = mesh_out_file

    inputs = image.squeeze(0).cpu()
    patch_setting1.save_inputs(ioh.in_dir, inputs, ioh.modelname, out_file_dict)

    # Copy to visualization directory for first vis_n_output samples
    c_it = model_counter[ioh.category_id]
    ioh.copy_to_vis(c_it, patch_setting1, out_file_dict)

    # increase model counter
    model_counter[ioh.category_id] += 1

# Create pandas dataframe and save
ioh.save_df(time_dicts)
