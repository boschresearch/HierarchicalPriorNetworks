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
import logging

import numpy as np
import open3d as o3d
import trimesh

from im2mesh.common import compute_iou
# from scipy.spatial import cKDTree
from im2mesh.utils.libkdtree import KDTree
from im2mesh.utils.libmesh import check_mesh_contains

# Maximum values for bounding box [-0.5, 0.5]^3
EMPTY_PCL_DICT = {
    'completeness': np.sqrt(3),
    'accuracy': np.sqrt(3),
    'completeness2': 3,
    'accuracy2': 3,
    'chamfer': 6,
}

EMPTY_PCL_DICT_NORMALS = {
    'normals completeness': -1.,
    'normals accuracy': -1.,
    'normals': -1.,
}

logger = logging.getLogger(__name__)


class MeshEvaluator(object):
    ''' Mesh evaluation class.

    It handles the mesh evaluation process.

    Args:
        n_points (int): number of points to be used for evaluation
    '''

    def __init__(self, n_points=100000):
        self.n_points = n_points

    def eval_mesh(self, mesh, pointcloud_tgt, normals_tgt,
                  points_iou, occ_tgt, pointcloud_tgt_visibility):
        ''' Evaluates a mesh.

        Args:
            mesh (trimesh): mesh which should be evaluated
            pointcloud_tgt (numpy array): target point cloud
            normals_tgt (numpy array): target normals
            points_iou (numpy_array): points tensor for IoU evaluation
            occ_tgt (numpy_array): GT occupancy values for IoU points
        '''
        if len(mesh.vertices) != 0 and len(mesh.faces) != 0:
            pointcloud, idx = mesh.sample(self.n_points, return_index=True)
            pointcloud = pointcloud.astype(np.float32)
            normals = mesh.face_normals[idx]
        else:
            pointcloud = np.empty((0, 3))
            normals = np.empty((0, 3))

        out_dict = self.eval_pointcloud(
            pointcloud, pointcloud_tgt,
            normals=normals, normals_tgt=normals_tgt,
            tgt_visibility=pointcloud_tgt_visibility
        )
        if len(mesh.vertices) != 0 and len(mesh.faces) != 0:
            occ = check_mesh_contains(mesh, points_iou)
            out_dict['iou'] = compute_iou(occ, occ_tgt)[0]
        else:
            out_dict['iou'] = 0.
        return out_dict

    def eval_pointcloud(self, pointcloud, pointcloud_tgt,
                        normals=None, normals_tgt=None, tgt_visibility=None):
        ''' Evaluates a point cloud.

        Args:
            pointcloud (numpy array): predicted point cloud
            pointcloud_tgt (numpy array): target point cloud
            normals (numpy array): predicted normals
            normals_tgt (numpy array): target normals
        '''
        # Return maximum losses if pointcloud is empty
        if pointcloud.shape[0] == 0:
            logger.warn('Empty pointcloud / mesh detected!')
            out_dict = EMPTY_PCL_DICT.copy()
            if normals is not None and normals_tgt is not None:
                out_dict.update(EMPTY_PCL_DICT_NORMALS)
            return out_dict

        pointcloud = np.asarray(pointcloud)
        pointcloud_tgt = np.asarray(pointcloud_tgt)

        # icp with o3d
        gt = o3d.geometry.PointCloud()
        gt.points = o3d.utility.Vector3dVector(pointcloud_tgt)
        pr = o3d.geometry.PointCloud()
        pr.points = o3d.utility.Vector3dVector(pointcloud)
        reg = o3d.registration.registration_icp(gt, pr, 0.01, np.identity(4),
                               o3d.registration.TransformationEstimationPointToPoint(with_scaling=True),
                               o3d.registration.ICPConvergenceCriteria(1e-6, 20))
        gt = gt.transform(reg.transformation)
        pointcloud_tgt = np.asarray(gt.points, dtype=np.float32)

        # Completeness: how far are the points of the target point cloud
        # from thre predicted point cloud
        completeness, completeness_normals, _ = distance_p2p(
            pointcloud_tgt, normals_tgt, pointcloud, normals
        )
        if tgt_visibility is not None:
            completeness_vis = completeness[tgt_visibility]
            completeness_invis = completeness[~tgt_visibility]

        # Accuracy: how far are th points of the predicted pointcloud
        # from the target pointcloud
        accuracy, accuracy_normals, closest_gt_idx = distance_p2p(
            pointcloud, normals, pointcloud_tgt, normals_tgt
        )
        if tgt_visibility is not None:
            pred_visibility = tgt_visibility[closest_gt_idx]
            accuracy_vis = accuracy[pred_visibility]
            accuracy_invis = accuracy[~pred_visibility]

        fscore_dict = {}
        for th_str, th in [("05", 1.0/200), ("1",1.0/100), ("2",1.0/50)]: # f@0.5 f@1 f@2
            recall = np.sum(completeness < th) / len(completeness)

            precision = np.sum(accuracy < th) / len(accuracy)
            if recall+precision > 0:
                fscore = 2 * recall * precision / (recall + precision)
            else:
                fscore = 0

            fscore_dict[f"fscore@{th_str}"] = fscore
            fscore_dict[f"recall@{th_str}"] = recall
            fscore_dict[f"precision@{th_str}"] = precision

            if tgt_visibility is not None:
                recall_vis = np.sum(completeness_vis < th) / len(completeness_vis)
                recall_invis = np.sum(completeness_invis < th) / len(completeness_invis)
                precision_vis = np.sum(accuracy_vis < th) / len(accuracy_vis)
                precision_invis = np.sum(accuracy_invis < th) / len(accuracy_invis)
                if recall_vis+precision_vis > 0:
                    fscore_vis = 2 * recall_vis * precision_vis / (recall_vis + precision_vis)
                else:
                    fscore_vis = 0
                if recall_invis+precision_invis > 0:
                    fscore_invis = 2 * recall_invis * precision_invis / (recall_invis + precision_invis)
                else:
                    fscore_invis = 0
                fscore_dict[f"fscore@{th_str}_vis"] = fscore_vis
                fscore_dict[f"fscore@{th_str}_invis"] = fscore_invis
                fscore_dict[f"recall@{th_str}_vis"] = recall_vis
                fscore_dict[f"recall@{th_str}_invis"] = recall_invis
                fscore_dict[f"precision@{th_str}_vis"] = precision_vis
                fscore_dict[f"precision@{th_str}_invis"] = precision_invis

        # sum them up
        completeness2 = (completeness**2).mean()
        completeness_normals = completeness_normals.mean()
        completeness = completeness.mean()

        accuracy2 = (accuracy**2).mean()
        accuracy_normals = accuracy_normals.mean()
        accuracy = accuracy.mean()
        chamfer = completeness + accuracy
        normals_correctness = (
            0.5 * completeness_normals + 0.5 * accuracy_normals
        )
        if tgt_visibility is not None:
            completeness2_vis = completeness_vis**2
            completeness2_invis = completeness_invis**2
            completeness2_vis = completeness2_vis.mean()
            completeness2_invis = completeness2_invis.mean()
            completeness_vis = completeness_vis.mean()
            completeness_invis = completeness_invis.mean()
            accuracy2_vis = accuracy_vis**2
            accuracy2_invis = accuracy_invis**2
            accuracy2_vis = accuracy2_vis.mean()
            accuracy2_invis = accuracy2_invis.mean()
            accuracy_vis = accuracy_vis.mean()
            accuracy_invis = accuracy_invis.mean()
            chamfer_vis = completeness_vis + accuracy_vis
            chamfer_invis = completeness_invis + accuracy_invis

        out_dict = {
            'chamfer': chamfer,
            'accuracy': accuracy,
            'accuracy2': accuracy2,
            'completeness': completeness,
            'completeness2': completeness2,
            'normals completeness': completeness_normals,
            'normals accuracy': accuracy_normals,
            'normals': normals_correctness,
        }
        if tgt_visibility is not None:
            out_dict_vis = {
                'chamfer_vis': chamfer_vis,
                'chamfer_invis': chamfer_invis,
                'accuracy_vis': accuracy_vis,
                'accuracy_invis': accuracy_invis,
                'accuracy2_vis': accuracy2_vis,
                'accuracy2_invis': accuracy2_invis,
                'completeness_vis': completeness_vis,
                'completeness_invis': completeness_invis,
                'completeness2_vis': completeness2_vis,
                'completeness2_invis': completeness2_invis,
            }
            out_dict.update(out_dict_vis)

        out_dict.update(fscore_dict)

        return out_dict


def distance_p2p(points_src, normals_src, points_tgt, normals_tgt):
    ''' Computes minimal distances of each point in points_src to points_tgt.

    Args:
        points_src (numpy array): source points
        normals_src (numpy array): source normals
        points_tgt (numpy array): target points
        normals_tgt (numpy array): target normals
    '''
    kdtree = KDTree(points_tgt)
    dist, idx = kdtree.query(points_src)

    if normals_src is not None and normals_tgt is not None:
        normals_src = \
            normals_src / np.linalg.norm(normals_src, axis=-1, keepdims=True)
        normals_tgt = \
            normals_tgt / np.linalg.norm(normals_tgt, axis=-1, keepdims=True)

        normals_dot_product = (normals_tgt[idx] * normals_src).sum(axis=-1)
        # Handle normals that point into wrong direction gracefully
        # (mostly due to mehtod not caring about this in generation)
        normals_dot_product = np.abs(normals_dot_product)
    else:
        normals_dot_product = np.array(
            [np.nan] * points_src.shape[0], dtype=np.float32)
    return dist, normals_dot_product, idx


def distance_p2m(points, mesh):
    ''' Compute minimal distances of each point in points to mesh.

    Args:
        points (numpy array): points array
        mesh (trimesh): mesh

    '''
    _, dist, _ = trimesh.proximity.closest_point(mesh, points)
    return dist
