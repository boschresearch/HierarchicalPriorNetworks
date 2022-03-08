""" visualize npz, ply, off files using Open3D"""
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

import sys
import os
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

def visualize(filepath):
    if not os.path.exists(filepath):
        print(f"File {filepath} does not exist.")
        exit(3)
    ext = filepath.split('.')[1]
    if ext not in ['npz', 'ply', 'off']:
        print(f"Extension '{ext}' unknown. Choose ['npz', 'ply', 'off'].")
        exit(3)

    if ext == 'npz':
        depth = np.load(filepath)['arr_0'][:,:,2]
        plt.imshow(depth)
        plt.show()
        # project as pointcloud
        print("Visualizing depth as Pointcloud")
        # Pseudo PointCloud: x,y,z
        height = depth.shape[0]
        width = depth.shape[1]
        x = list(range(0, width)) * height
        y = [j for j in range(0, height) for i in range(0, width)]
        z = depth.reshape(-1)
        # multiply the depth value by approximately the image width
        # s.t. the values are in the same range
        z *= width
        points = np.asarray([x, y, z]).transpose()
        points -= np.mean(points, axis=0)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        # account for open3d coordinate system with y pointing up and z pointing out of screen
        pcd = pcd.rotate(np.asarray([3.1416,0,0]), center=False) # rotate around x
        o3d.visualization.draw_geometries([pcd])

    if ext in ['ply', 'off']:
        mesh = o3d.io.read_triangle_mesh(filepath)
        o3d.visualization.draw_geometries([mesh])

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Call with path to file to visualize.")
    visualize(sys.argv[1])