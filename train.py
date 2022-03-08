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

import matplotlib;
import numpy as np
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter

matplotlib.use('Agg')
from im2mesh import config, data
from im2mesh.checkpoints import CheckpointIO

# Arguments
parser = argparse.ArgumentParser(
    description='Train a 3D reconstruction model.'
)
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
parser.add_argument('--exit-after', type=int, default=-1,
                    help='Checkpoint and exit after specified number of seconds'
                         'with exit code 2.')
parser.add_argument('--exit-after-epochs', type=int, default=-1,
                    help='Checkpoint and exit after specified number of epochs')
parser.add_argument('--exit-after-it', type=int, default=-1,
                    help='Checkpoint and exit after specified number of iterations')
parser.add_argument('--exit-threshold', type=float, default=-1,
                    help='Checkpoint and exit when relative threshold is reached')

args = parser.parse_args()
cfg = config.load_config(args.config, 'configs/default.yaml')
is_cuda = (torch.cuda.is_available() and not args.no_cuda)
device = torch.device("cuda" if is_cuda else "cpu")

# Set t0
t0 = time.time()

# Shorthands
out_dir = cfg['training']['out_dir']
batch_size = cfg['training']['batch_size']
batch_size_val = cfg['training']['batch_size_val']
batch_size_vis = cfg['training']['batch_size_vis']
backup_every = cfg['training']['backup_every']
exit_after = args.exit_after

model_selection_metric = cfg['training']['model_selection_metric']
if cfg['training']['model_selection_mode'] == 'maximize':
    model_selection_sign = 1
elif cfg['training']['model_selection_mode'] == 'minimize':
    model_selection_sign = -1
else:
    raise ValueError('model_selection_mode must be '
                     'either maximize or minimize.')

# Output directory
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# Dataset
train_dataset = config.get_dataset('train', cfg)
val_dataset = config.get_dataset('val', cfg)

# TODO: clarify nodes == workers?
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, num_workers=4, shuffle=True,
    collate_fn=data.collate_remove_none,
    worker_init_fn=data.worker_init_fn)

# debug the loading procedure
# for elt in train_loader:
#     print(elt['points'])
#     print(f"size of train ds elt  {sys.getsizeof(elt)}")
#     print(f"size of train ds points  {sys.getsizeof(elt['points'])}")
#     print(f"size of train ds occ  {sys.getsizeof(elt['points.occ'])}")
#     print(f"size of train ds inputs  {sys.getsizeof(elt['inputs'])}")
# train_loader = list(iter(train_loader)) # load all into memory
# print(deep_getsizeof(train_loader, set()))
# print(f"size of train ds elt  {sys.getsizeof(train_loader[0])}")
# print(f"size of train ds list {sys.getsizeof(train_loader)}")
# exit()
# for i, elt in enumerate(train_loader):
#     # elt = train_loader[0]
#     points = elt["points"].numpy()
#     occ = elt["points.occ"].numpy()
#     occ = occ.astype(np.bool)
#     points = np.squeeze(points)
#     occ = np.squeeze(occ)
#     print(np.sum(occ))
#     print(occ.shape)
#     inputs = elt["inputs"].numpy()
#     inputs = np.squeeze(inputs)
#     # np.save(f"part_{i}.npy", points[occ])
#     # np.savez(f"part_{i}.npz", points=points)
#     np.savez(f"part_{i}.npz", points=points[occ])
#     matplotlib.pyplot.imsave(f"part_{i}.jpg", inputs)
# exit()
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size_val, num_workers=4, shuffle=False,
    collate_fn=data.collate_remove_none,
    worker_init_fn=data.worker_init_fn)
# val_loader = list(iter(val_loader)) # load all into memory

# For visualizations
vis_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size_vis, shuffle=True,
    collate_fn=data.collate_remove_none,
    worker_init_fn=data.worker_init_fn)
data_vis = next(iter(vis_loader))
while data_vis == None:
    # print(type(data_vis))
    data_vis = next(iter(vis_loader))

# Model
model = config.get_model(cfg, device=device, dataset=train_dataset)

# Intialize training
npoints = 1000
optimizer = optim.Adam(model.parameters(), lr=1e-4)
# optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
trainer = config.get_trainer(model, optimizer, cfg, device=device)

checkpoint_io = CheckpointIO(out_dir, model=model, optimizer=optimizer)
try:
    load_dict = checkpoint_io.load('model.pt')
except FileExistsError:
    load_dict = dict()
epoch_it = load_dict.get('epoch_it', -1)
it = load_dict.get('it', -1)
metric_val_best = load_dict.get(
    'loss_val_best', -model_selection_sign * np.inf)

# Hack because of previous bug in code
# TODO: remove, because shouldn't be necessary
if metric_val_best == np.inf or metric_val_best == -np.inf:
    metric_val_best = -model_selection_sign * np.inf

# TODO: remove this switch
# metric_val_best = -model_selection_sign * np.inf

print('Current best validation metric (%s): %.8f'
      % (model_selection_metric, metric_val_best))

# TODO: reintroduce or remove scheduler?
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4000,
#                                       gamma=0.1, last_epoch=epoch_it)
logger = SummaryWriter(os.path.join(out_dir, 'logs'))

# Shorthands
print_every = cfg['training']['print_every']
checkpoint_every = cfg['training']['checkpoint_every']
validate_every = cfg['training']['validate_every']
visualize_every = cfg['training']['visualize_every']

# Print model
nparameters = sum(p.numel() for p in model.parameters())
# print(model)
print('Total number of parameters: %d' % nparameters)

while True:
    epoch_it += 1
#     scheduler.step()
    # set epoch start time
    t_ep = time.time()
    # init iteration timer
    t_it = 0
    t_it_fullcycle = time.time()
    # print(len(train_loader))

    for batch in train_loader:
        t_it_start = time.time() # start time for one iteration
        # print(type(batch))
        if batch == None:
            continue
        if batch['inputs'].shape[0] == 1: # during training no batch size of 1
            # print(batch['inputs'].shape)
            continue
        it += 1
        loss = trainer.train_step(batch)
        logger.add_scalar('train/loss', loss, it)

        # log time needed for one iteration and sum up
        t_it += time.time() - t_it_start

        # needed when switching from loss to iou
        # (loss will be much larger than 1, iou max is one)
        # if metric_val_best > 1:
        #     metric_val_best = 0.0

        # Print output
        if print_every > 0 and (it % print_every) == 0:
            # full cycle time, this -unlike t_it- includes train_loader
            t_full = time.time() - t_it_fullcycle
            t_it_fullcycle = time.time()
            print('[Epoch %02d] it=%03d, t_train=%.2fs, t_full=%.1fs, loss=%.4f'
                  % (epoch_it, it, t_it, t_full, loss))
            t_it = 0 # reset logging iteration timer

        # Visualize output
        if visualize_every > 0 and (it % visualize_every) == 0:
            print('Visualizing')
            trainer.visualize(data_vis)

        # Save checkpoint
        if (checkpoint_every > 0 and (it % checkpoint_every) == 0):
            print('Saving checkpoint')
            checkpoint_io.save('model.pt', epoch_it=epoch_it, it=it,
                               loss_val_best=metric_val_best)

        # Backup if necessary
        if (backup_every > 0 and (it % backup_every) == 0):
            print('Backup checkpoint')
            checkpoint_io.save('model_%d.pt' % it, epoch_it=epoch_it, it=it,
                               loss_val_best=metric_val_best)
        # Run validation
        if validate_every > 0 and (it % validate_every) == 0:
            eval_dict = trainer.evaluate(val_loader)
            metric_val = eval_dict[model_selection_metric]
            print('Validation metric (%s): %.4f'
                  % (model_selection_metric, metric_val))

            for k, v in eval_dict.items():
                logger.add_scalar('val/%s' % k, v, it)

            # TODO: improve exit threshold computation
            diff = abs(metric_val - metric_val_best)
            if model_selection_sign * (metric_val - metric_val_best) > 0:
                metric_val_best = metric_val
                print('New best model (loss %.4f)' % metric_val_best)
                checkpoint_io.save('model_best.pt', epoch_it=epoch_it, it=it,
                                   loss_val_best=metric_val_best)
            # stop if diff to prev best is small
            if args.exit_threshold > 0 and diff < args.exit_threshold:
                print(f'Relative threshold {args.exit_threshold} reached. Exiting.')
                checkpoint_io.save('model.pt', epoch_it=epoch_it, it=it,
                                   loss_val_best=metric_val_best)
                exit(3)

        # Exit if necessary
        if exit_after > 0 and (time.time() - t0) >= exit_after:
            print('Time limit reached. Exiting.')
            checkpoint_io.save('model.pt', epoch_it=epoch_it, it=it,
                               loss_val_best=metric_val_best)
            exit(3)
        # Exit after epochs
        if args.exit_after_epochs > 0 and epoch_it >= args.exit_after_epochs:
            print('Epoch limit reached. Exiting.')
            checkpoint_io.save('model_end.pt', epoch_it=epoch_it, it=it,
                               loss_val_best=metric_val_best)
            exit(3)
        if args.exit_after_it > 0 and it >= args.exit_after_it:
            print('Epoch limit reached. Exiting.')
            checkpoint_io.save('model_end.pt', epoch_it=epoch_it, it=it,
                               loss_val_best=metric_val_best)
            exit(3)
    print(f"Time per [{epoch_it:4}] epoch: {(time.time()-t_ep):.1f}s - it [{it}]")