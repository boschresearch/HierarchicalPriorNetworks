# This file contains the functions to train the model
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

import os

import torch
from torch import distributions as dist
from torch.nn import functional as F
from tqdm import trange

from im2mesh.common import (
    compute_iou, make_3d_grid
)
from im2mesh.training import BaseTrainer
from im2mesh.utils import visualize as vis


class Trainer(BaseTrainer):
    ''' Trainer object for the Occupancy Network.

    Args:
        model (nn.Module): Occupancy Network model
        optimizer (optimizer): pytorch optimizer object
        device (device): pytorch device
        input_type (str): input type
        vis_dir (str): visualization directory
        threshold (float): threshold value
        eval_sample (bool): whether to evaluate samples

    '''

    def __init__(self, model, optimizer, device=None, input_type='img',
                 vis_dir=None, threshold=0.5, eval_sample=False):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.input_type = input_type
        self.vis_dir = vis_dir
        self.threshold = threshold
        self.eval_sample = eval_sample

        if vis_dir is not None and not os.path.exists(vis_dir):
            os.makedirs(vis_dir)

    def train_step(self, data):
        ''' Performs a training step.

        Args:
            data (dict): data dictionary
        '''
        self.model.train()
        self.optimizer.zero_grad()
        loss = self.compute_loss(data)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def eval_step(self, data):
        ''' Performs an evaluation step.

        Args:
            data (dict): data dictionary
        '''
        self.model.eval()

        device = self.device
        threshold = self.threshold
        eval_dict = {}

        # Compute elbo
        points = data.get('points').to(device)
        occ = data.get('points.occ').to(device)
        sdf = data.get('points.sdf').to(device)

        inputs = data.get('inputs', torch.empty(points.size(0), 0)).to(device)
        voxels_occ = data.get('voxels')

        points_iou = data.get('points_iou').to(device)
        occ_iou = data.get('points_iou.occ').to(device)
        valid_iou = data.get('points_iou.valid').cpu().numpy()
        # sdf_iou = data.get('points_iou.sdf').to(device)

        # TODO include L1 distance between gt and predicted sdf
        # sdf_l1 = data.get('points_iou.sdf').to(device)
        # TODO load and copy sdf in core (point_iou hack)

        kwargs = {}

        with torch.no_grad():
            elbo, rec_error, kl = self.model.compute_elbo(
                points, sdf, inputs, **kwargs)

        eval_dict['loss'] = -elbo.mean().item()
        eval_dict['rec_error'] = rec_error.mean().item()
        # eval_dict['kl'] = kl.mean().item()

        # Compute iou
        batch_size = points.size(0)

        with torch.no_grad():
            p_out = self.model(points_iou, inputs,
                               sample=self.eval_sample, **kwargs)

        occ_iou_np = (occ_iou >= 0.5).cpu().numpy()
        # threshold is 0 and negative sdf values mean we are inside the model
        occ_iou_hat_np = (p_out <= threshold).cpu().numpy()
        # iou = compute_iou(occ_iou_np, occ_iou_hat_np).mean()
        # remove results for padded points
        iou = compute_iou(occ_iou_np[valid_iou], occ_iou_hat_np[valid_iou]).mean()
        eval_dict['iou'] = iou

        # Estimate voxel iou
        if voxels_occ is not None:
            voxels_occ = voxels_occ.to(device)
            points_voxels = make_3d_grid(
                (-0.5 + 1/64,) * 3, (0.5 - 1/64,) * 3, (32,) * 3)
            points_voxels = points_voxels.expand(
                batch_size, *points_voxels.size())
            points_voxels = points_voxels.to(device)
            with torch.no_grad():
                p_out = self.model(points_voxels, inputs,
                                   sample=self.eval_sample, **kwargs)

            voxels_occ_np = (voxels_occ >= 0.5).cpu().numpy()
            # same as for point_iou
            occ_hat_np = (p_out <= threshold).cpu().numpy()
            iou_voxels = compute_iou(voxels_occ_np, occ_hat_np).mean()

            eval_dict['iou_voxels'] = iou_voxels

        return eval_dict

    def visualize(self, data):
        ''' Performs a visualization step for the data.

        Args:
            data (dict): data dictionary
        '''
        device = self.device

        batch_size = data['points'].size(0)
        inputs = data.get('inputs', torch.empty(batch_size, 0)).to(device)

        shape = (32, 32, 32)
        p = make_3d_grid([-0.5] * 3, [0.5] * 3, shape).to(device)
        p = p.expand(batch_size, *p.size())

        kwargs = {}
        with torch.no_grad():
            logits = self.model(p, inputs, sample=self.eval_sample, **kwargs)

        occ_hat = logits.view(batch_size, *shape)
        voxels_out = (occ_hat <= self.threshold).cpu().numpy()


        for i in trange(batch_size):
            input_img_path = os.path.join(self.vis_dir, '%03d_in.png' % i)
            vis.visualize_data(
                inputs[i].cpu(), self.input_type, input_img_path)
            vis.visualize_voxels(
                voxels_out[i], os.path.join(self.vis_dir, '%03d.png' % i))

    def compute_loss(self, data):
        ''' Computes the loss.

        Args:
            data (dict): data dictionary
        '''
        device = self.device
        p = data.get('points').to(device)
        occ = data.get('points.occ').to(device)
        gt_sdf = data.get('points.sdf').to(device)
        inputs = data.get('inputs', torch.empty(p.size(0), 0)).to(device)

        kwargs = {}

        c = self.model.encode_inputs(inputs)
        q_z = self.model.infer_z(p, occ, c, **kwargs)
        z = q_z.rsample()

        # KL-divergence
        kl = dist.kl_divergence(q_z, self.model.p0_z).sum(dim=-1)
        loss = kl.mean()

        # General points
        logits = self.model.decode(p, z, c, **kwargs) # removed bernoulli

        # mask padded points
        valid_mask = data.get('points.valid').to(device)
        logits = logits[valid_mask]
        occ = occ[valid_mask]

        # deepsdf
        # last operation in decoder is tanh
        # pred_sdf = decoder(input)
        # chunk_loss = loss_l1(pred_sdf, sdf_gt[i].cuda()) / num_sdf_samples

        # sdfnet
        # https://github.com/devlearning-gt/3DShapeGen/blob/9dd38f92f7fda1b1538fd9067b3196afba43d7a9/SDFNet/utils.py#L181

        # L1 distance
        loss_i = F.l1_loss(logits, gt_sdf, reduction='none')
        # sdfnet weighting
        thres = 0.01
        weight = 4.0
        weight_mask = torch.ones(loss_i.shape).to(device)
        weight_mask[torch.abs(gt_sdf) < thres] =\
            weight_mask[torch.abs(gt_sdf) < thres]*weight
        loss_i = loss_i * weight_mask

        # weight
        # loss_i = F.binary_cross_entropy_with_logits(
        #     logits, occ, weight=distance_weight, reduction='none')
        # no weight
        # loss_i = F.binary_cross_entropy_with_logits(
        #     logits, occ, reduction='none')

        batchsize = valid_mask.shape[0]
        loss = loss + (loss_i.sum(-1) / batchsize) # .mean()
        # loss = loss + loss_i.sum(-1).mean()

        return loss
