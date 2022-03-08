""" render a depth image from ShapeNet models using PyTorch3D"""
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
import torch

# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes, load_obj
from pytorch3d.transforms import (
    Transform3d,
    RotateAxisAngle,
)
# from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    FoVOrthographicCameras,
    PointLights,
    DirectionalLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesUV,
    Textures,
    TexturesVertex)

import matplotlib.pyplot as plt
import numpy as np

# Setup
on_gpu = (torch.cuda.is_available() and True)
device = torch.device("cuda" if on_gpu else "cpu")

def render_depth(mesh, raster):
    fragments = raster(mesh)
    depth_img = fragments.zbuf[:, :, :, 0].cpu().numpy().squeeze()
    # show vertex indices
    # plt.imshow(fragments.pix_to_face.cpu().numpy().squeeze())
    # plt.grid("off");
    # plt.axis("off");
    # plt.show()
    return depth_img

def get_camera(elev, azim, dist=1.0, perspective=True, **cam_kwargs):
    # Initialize a camera.
    # With world coordinates +Y up, +X left and +Z in, the front of the cow is facing the -Z direction.
    # So we move the camera by 180 in the azimuth direction so it is facing the front of the cow.
    R, T = look_at_view_transform(dist, elev, azim) # , at=((0,-0.2,0),)
    if perspective:
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
    else:
        cameras = FoVOrthographicCameras(device=device, R=R, T=T, **cam_kwargs)
    return cameras

def center_vertices(vertices):
    """ shift the center of the mesh BB / vertices to the origin """
    max_vert = vertices.min(dim=0).values
    min_vert = vertices.max(dim=0).values
    # print(f"Translation {- min_vert - ((max_vert - min_vert) / 2)}")
    vertices = vertices - min_vert - ((max_vert - min_vert) / 2)
    return vertices

def normalize_vertices(vertices):
    min_v = vertices.min(dim=0).values
    max_v = vertices.max(dim=0).values
    largest_extend = max(max_v - min_v)
    # print(f"Normalize: t {-min_v} scale {1/largest_extend}")
    return (vertices - min_v) / largest_extend

def render_orthographic_depth(args):
    os.makedirs(args.out_dir, exist_ok=True)

    # get info from xml
    classid, modelid = args.filename.split('_')
    if os.path.exists(os.path.join(args.out_dir, args.filename + ".npz")):
        print(f"File {args.out_dir} {args.filename} exists. Skip rendering.")
        exit(3)

    obj_filename = os.path.join(args.shapenet_dir, classid, modelid, "models/model_normalized.obj")
    # print(obj_filename)

    # Load obj file
    verts, faces, aux = load_obj(obj_filename, load_textures=False, device=device)
    mesh_vertices = verts
    mesh_vertices = center_vertices(mesh_vertices)

    # rotate mesh into view and normalize
    azimuth = RotateAxisAngle(args.azimuth, axis="Y", device=device)
    elevation = RotateAxisAngle(-args.elevation, axis="X", device=device)
    new_vertices = Transform3d(device=device).compose(azimuth,elevation).transform_points(mesh_vertices)

    # normalize mesh in new rotation
    new_vertices = normalize_vertices(new_vertices)
    # shift to -0.5 # do not shift! do centering instead!

    # center again
    new_vertices = center_vertices(new_vertices)
    mesh = Meshes(verts=[new_vertices], faces=[faces.verts_idx])
    # mesh = origmesh
    mesh = mesh.to(device)

    # RENDERING
    # set cam slightly away from model min
    cameras = get_camera(0, 0, dist=0.50001, perspective=False, scale_xyz=((2,2,2),))
    # Refer to rasterize_meshes.py for explanations of these parameters.
    # Refer to docs/notes/renderer.md for an explanation of
    # the difference between naive and coarse-to-fine rasterization.
    raster_settings = RasterizationSettings(
        image_size=256,
        blur_radius=0.0,
        faces_per_pixel=1,
        max_faces_per_bin=50000, # fix the overflow issue
    )
    raster = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
    depth = render_depth(mesh, raster)

    # WRITE DEPTH IMAGES
    depth = np.stack([depth, depth, depth], axis=2)
    np.savez(os.path.join(args.out_dir, args.filename + ".npz"), depth)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Render settings.')
    parser.add_argument('--filename', type=str, default='02691156_ffc1b82bec23a50995b8d6bdd18d56e8',
                        help='classid_modelid')
    parser.add_argument('--azimuth', type=float, default=10, help='Rotate the model into view. Rotation around Y in degrees.')
    parser.add_argument('--elevation', type=float, default=10, help='Rotate the model into view. Rotation around X in degrees.')
    parser.add_argument('--out_dir', type=str, default='data/renders/', help='where to save rendered images')
    parser.add_argument('--shapenet_dir', type=str, default='data/ShapeNetCore.v2/', help='Path to ShapeNetCore.v2')
    args = parser.parse_args()
    render_orthographic_depth(args)
