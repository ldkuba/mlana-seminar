import math
import torch
import numpy as np
from torch import nn, Tensor
import auto_encoder as ae

import trimesh

if __name__ == "__main__":
    # parameters
    

    # load the mesh
    mesh = trimesh.load("lantern.obj")
    vertices = torch.tensor(mesh.vertices, dtype=torch.float32)
    faces = torch.tensor(mesh.faces, dtype=torch.int64)

    # normalize the mesh to [-1, 1]
    aabb = mesh.bounds.astype(np.float32)
    center = (aabb[0] + aabb[1]) / 2
    scale = max(aabb[1] - aabb[0])
    vertices = (vertices - center) / scale

    # Testing
    # vertices = torch.tensor([[0.2, 0.4, 0.3],[0.5, 0.2, 0.8],[0.8, 0.8, 0.1],[0.3, 0.9, 0.7],[1.5, 1.6, 1.8],[1.3, 1.8, 2.0]], dtype=torch.float32)
    # faces = torch.tensor([[0, 1, 5], [3, 1, 2], [2, 4, 5], [5, 4, 0], [0, 4, 2]], dtype=torch.int64)

    autoEnc = ae.AutoEncoder() 
    result = autoEnc(ae.AutoEncodeData(vertices, faces))
    print(result.shape)
