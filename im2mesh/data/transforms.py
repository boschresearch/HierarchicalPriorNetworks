# This file contains the transforms applied to data, including the local crop
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

from im2mesh.common import get_crop_mask


# Transforms
class PointcloudNoise(object):
    ''' Point cloud noise transformation class.

    It adds noise to point cloud data.

    Args:
        stddev (int): standard deviation
    '''

    def __init__(self, stddev):
        self.stddev = stddev

    def __call__(self, data):
        ''' Calls the transformation.

        Args:
            data (dictionary): data dictionary
        '''
        data_out = data.copy()
        points = data[None]
        noise = self.stddev * np.random.randn(*points.shape)
        noise = noise.astype(np.float32)
        data_out[None] = points + noise
        return data_out


class SubsamplePointcloud(object):
    ''' Point cloud subsampling transformation class.

    It subsamples the point cloud data.

    Args:
        N (int): number of points to be subsampled
    '''
    def __init__(self, N):
        self.N = N

    def __call__(self, data):
        ''' Calls the transformation.

        Args:
            data (dict): data dictionary
        '''
        data_out = data.copy()
        points = data[None]
        normals = data['normals']

        indices = np.random.randint(points.shape[0], size=self.N)
        data_out[None] = points[indices, :]
        data_out['normals'] = normals[indices, :]

        return data_out

class NoTransform(object):
    ''' Does not do anything
    '''
    def __init__(self):
        pass

    def __call__(self, data):
        ''' Calls the transformation.

        Args:
            data (dict): data dictionary
        '''
        return data

class SubsamplePoints(object):
    ''' Points subsampling transformation class.

    It subsamples the points data.

    Args:
        N (int): number of points to be subsampled
    '''
    def __init__(self, N, mode):
        self.N = N
        self.mode = mode

    def __call__(self, data):
        ''' Calls the transformation.

        Args:
            data (dictionary): data dictionary
        '''
        # only subsample training points (not points_iou),
        # during test data does not contain points
        # data is None if img patch is black
        if self.mode == 'test' or data is None:
            return data

        points = data['points']
        occ = data['points.occ']
        sdf = data['points.sdf']
        valid = data['points.valid']

        data_out = data.copy()
        if isinstance(self.N, int):
            idx = np.random.randint(points.shape[0], size=self.N)
            data_out.update({
                'points': points[idx, :],
                'points.occ':  occ[idx],
                'points.sdf': sdf[idx],
                'points.valid': valid[idx],
                # don't do this here
                # 'sdf': sdf[idx] / sdf[idx].sum(), # normalize subsample
            })
        else:
            Nt_out, Nt_in = self.N
            occ_binary = (occ >= 0.5)
            points0 = points[~occ_binary]
            points1 = points[occ_binary]

            idx0 = np.random.randint(points0.shape[0], size=Nt_out)
            idx1 = np.random.randint(points1.shape[0], size=Nt_in)

            points0 = points0[idx0, :]
            points1 = points1[idx1, :]
            points = np.concatenate([points0, points1], axis=0)

            occ0 = np.zeros(Nt_out, dtype=np.float32)
            occ1 = np.ones(Nt_in, dtype=np.float32)
            occ = np.concatenate([occ0, occ1], axis=0)

            volume = occ_binary.sum() / len(occ_binary)
            volume = volume.astype(np.float32)

            sdf0 = sdf[~occ_binary]
            sdf1 = sdf[occ_binary]
            sdf0 = sdf0[idx0]
            sdf1 = sdf1[idx1]
            sdf = np.concatenate([sdf0, sdf1], axis=0)
            # don't do this here
            # sdf /= sdf.sum() # normalize subsample

            valid0 = valid[~occ_binary]
            valid1 = valid[occ_binary]
            valid0 = valid0[idx0]
            valid1 = valid1[idx1]
            valid = np.concatenate([valid0, valid1], axis=0)

            data_out.update({
                'points': points,
                'points.occ': occ,
                'points.sdf': sdf,
                'points.valid': valid,
            })
        return data_out


class PaddPoints(object):
    ''' Points padding transformation class.

    It padds the points data in case parts have too few.

    Args:
        N (int): number of points that the returned data should have
    '''
    def __init__(self, N):
        self.N = N

    def __call__(self, data):
        ''' Calls the transformation.

        Args:
            data (dictionary): data dictionary
        '''
        points = data[None]
        occ = data['occ']
        sdf = data['sdf']

        if self.N <= points.shape[0]:
            # always have 'valid' in the data dict, to have a consistent batch
            data.update({'valid':np.ones((points.shape[0]), dtype=np.bool)})
            return data

        data_out = data.copy()
        if isinstance(self.N, int):
            padd_points = np.zeros((self.N, points.shape[1]), dtype=points.dtype) - 0.5
            valid = np.zeros((self.N), dtype=np.bool)
            valid[:points.shape[0]] = True
            padd_points[:points.shape[0],:] = points
            padd_occ = np.zeros((self.N), dtype=occ.dtype)
            padd_occ[:occ.shape[0]] = occ
            padd_sdf = np.zeros((self.N), dtype=sdf.dtype) -1 # 0 sdf means on surface...
            padd_sdf[:sdf.shape[0]] = sdf
            data_out.update({
                None: padd_points,
                'occ':  padd_occ,
                'sdf': padd_sdf,
                'valid': valid,
                # don't do this here
                # 'sdf': sdf[idx] / sdf[idx].sum(), # normalize subsample
            })
        return data_out


class CropPatchAndPart(object):
    ''' Crop patch from image and part from 3D GT transformation class.

    It determines the image and cube indices. Unpacks the data dict.
    Crops a part for a given image

    Args:
        N (int): number of points to be subsampled
    '''
    def __init__(self, cfg, mode):
        self.cfg = cfg
        self.mode = mode

    def crop_patch(self, image, img_part_idx):
        """ take image and crop image patch"""
        img_part = image[
            :,
            img_part_idx[0]:img_part_idx[2],
            img_part_idx[1]:img_part_idx[3],
        ]
        data = {"inputs": img_part}
        return data

    def crop_part(self, points, occ, sdf, cube_part_idx):
        """ """
        # cube_part_idx = self.cube_indices[index]
        mask = get_crop_mask(points, cube_part_idx[0], cube_part_idx[1])
        # extract part
        points = points[mask]
        occupancies = occ[mask]
        # TODO for parts, the sdf value should be recomputed, might be that closest shape is in next part
        sdf = sdf[mask]

        # for local unit cube reconstruction transform coordinates
        # rescale all dimensions to -0.5,0.5, i.e. change aspect ratio
        minp = np.min(points, axis=0)
        maxp = np.max(points, axis=0)
        points = ((points - minp) / (maxp - minp)) - 0.5

        # # or rescale to cube dimensions, experiments don't show a difference
        # cube_min = cube_part_idx[0].astype(np.float32)
        # cube_max = cube_part_idx[1].astype(np.float32)
        # # these two variants might actually be the same, since (empty) points are all over the place
        # points = ((points - cube_min) / (cube_max - cube_min)) - 0.5

        data = {
            None: points,
            'occ': occupancies,
            'sdf': sdf,
        }
        return data

    def __call__(self, data):
        ''' Calls the transformation to crop patches.

        Args:
            data (dictionary): data dictionary
            random (bool):  True - training - return a random image patch
                            False - test - return all patches after another (internal counter)
        '''
        image = data["inputs"]

        if self.mode != 'test':
            points = data["points"]
            occ = data['points.occ']
            sdf = data['points.sdf']
        if self.mode == 'val':
            points_iou = data['points_iou'] # only in val dict?
            points_iou_occ = data['points_iou.occ']
            points_iou_sdf = data['points_iou.sdf']

        # Flaws
        # 1: improve test / train / val modi via separate calls
        # 3: make transforms separate - not for padding because of dict naming

        img_part_idx = data['img_patch_idx']
        cube_part_idx = data['cube_part_idx']

        image_data  = self.crop_patch(image, img_part_idx)
        if self.mode != 'test':
            points_data = self.crop_part(points, occ, sdf, cube_part_idx)
        if self.mode == 'val':
            points_iou_data = self.crop_part(points_iou,
                                             points_iou_occ,
                                             points_iou_sdf,
                                             cube_part_idx)

        # return none if part is black and we are training
        if (image_data["inputs"] >=0).sum() == 0 and (self.mode in ["train", "val"]):
            return None
        if self.mode != 'test':
            if points_data["occ"].sum() == 0:
                return None

        if self.mode == 'val':
            padd_points_transform = PaddPoints(100000) # points iou with 100k for validation
            points_iou_padd = padd_points_transform(points_iou_data)
        # In case parts are too small and have less points than subsample -> padd
        if self.mode != 'test':
            padd_points_transform = PaddPoints(self.cfg['data']['points_subsample'])
            points_padd = padd_points_transform(points_data)

        # fuse points into points.None and points_iou.None again
        if self.mode == 'val':
            name_list = ["points", "points_iou"]
            dict_list = [points_padd, points_iou_padd]
        elif self.mode =='train':
            name_list = ["points"]
            dict_list = [points_padd]
        else:
            name_list = []
            dict_list = []
        points_out_data = {}
        for field_name, field_dict in zip(name_list, dict_list):
            for k, v in field_dict.items():
                if k is None:
                    points_out_data[field_name] = v
                else:
                    points_out_data['%s.%s' % (field_name, k)] = v

        data_out = data.copy()
        data_out.update(image_data)
        data_out.update(points_out_data)
        return data_out
