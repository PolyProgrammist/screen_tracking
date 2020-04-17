import os
import torch
import numpy as np
from tqdm import tqdm_notebook
import imageio
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from skimage import img_as_ubyte

# io utils
from pytorch3d.io import load_obj

# datastructures
from pytorch3d.structures import Meshes, Textures

# 3D transformations functions
from pytorch3d.transforms import Rotate, Translate

# rendering components
from pytorch3d.renderer import (
    OpenGLPerspectiveCameras, look_at_view_transform, look_at_rotation,
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, HardPhongShader, PointLights
)
from pytorch3d.renderer import (
    look_at_view_transform,
    OpenGLPerspectiveCameras,
    PointLights,
    DirectionalLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    TexturedSoftPhongShader,
    SoftSilhouetteShader
)

# Set the cuda device
device = torch.device("cuda:0")
torch.cuda.set_device(device)

# Load the obj and ignore the textures and materials.
# verts, faces_idx, _ = load_obj("./data/teapot.obj")
# faces = faces_idx.verts_idx

# # Initialize each vertex to be white in color.
# verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
# textures = Textures(verts_rgb=verts_rgb.to(device))

# # Create a Meshes object for the teapot. Here we have only one mesh in the batch.
# teapot_mesh = Meshes(
#     verts=[verts.to(device)],
#     faces=[faces.to(device)],
#     textures=textures
# )

myimg = imageio.imread('doska.jpg')
myimg = np.array(myimg) / 255
myimg = myimg.astype(np.float32)
timg = torch.from_numpy(myimg)
mesh_img = timg.to(device)[None]


obj_filename = './tv_picture_centered.obj'
obj_filename = './card_100x100.obj'
verts, faces_idx, aux = load_obj(obj_filename)
faces = faces_idx.verts_idx
verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
verts_rgb[:,:,2] = 0

faces_torch = torch.from_numpy(np.array([np.array(faces_idx.textures_idx)]))
verts_uvs = torch.from_numpy(np.array([np.array(aux.verts_uvs)]))

print(faces_torch)
print(verts_uvs)


textures = Textures(maps=mesh_img, verts_uvs=verts_uvs.to(device), faces_uvs=faces_torch.to(device),
                    verts_rgb=verts_rgb.to(device))

# Create a Meshes object for the teapot. Here we have only one mesh in the batch.
teapot_mesh = Meshes(
    verts=[verts.to(device)],
    faces=[faces.to(device)],
    textures=textures
)