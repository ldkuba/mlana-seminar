import math
import torch
import numpy as np
from torch import nn, Tensor
import trimesh.viewer
import auto_encoder as ae
import graph_encoder as ge

import time

import trimesh

def discretize(x, min_val, max_val, num_values):
    x = (x - min_val) / (max_val - min_val)
    x = x * (num_values - 1)
    x = torch.round(x).long()
    return x

num_discrete_values = 128

# Loads the model and returns the sorted vertices, faces and edge list
def load_model(input_file) -> ae.AutoEncodeData:
    # load the mesh
    mesh = trimesh.load(input_file)
    vertices = torch.tensor(mesh.vertices, dtype=torch.float32)
    faces = torch.tensor(mesh.faces, dtype=torch.int64)

    # normalize the mesh to [-1, 1]
    aabb = mesh.bounds.astype(np.float32)
    center = (aabb[0] + aabb[1]) / 2
    scale = max(aabb[1] - aabb[0])
    vertices = (vertices - center) / scale

    with torch.device("cuda"):
        # sort the mesh vertices and faces
        vertices_sorted, faces_sorted = ge.sort_mesh(vertices.to('cuda'), faces.to('cuda'))
        # compute edge list
        edge_list = ge.compute_edge_list(faces_sorted)

        # compute angles, face areas and normals
        angles, face_areas, normals = ge.compute_angles_areas_normals(vertices_sorted, faces_sorted)

        # distribute the vertex data to the faces
        face_vertices = vertices_sorted[faces_sorted, :].flatten(1)

        # quantize face features
        vertex_discrete = discretize(face_vertices, -1, 1, num_discrete_values)
        angles_discrete = discretize(angles, 0, math.pi, num_discrete_values)
        areas_discrete = discretize(face_areas, 0, 4, num_discrete_values)
        normals_discrete = discretize(normals, -1, 1, num_discrete_values)

        return ae.AutoEncodeData(vertex_discrete, vertices_sorted.size(0), faces_sorted, angles_discrete, areas_discrete, normals_discrete, edge_list)

if __name__ == "__main__":
    # parameters
    lr = 1e-1

    input_data = load_model("lantern.obj")

    with torch.device("cuda"):
        autoEnc = ae.AutoEncoder().to("cuda")
        optimizer = torch.optim.Adam(autoEnc.parameters(), lr=lr)

        # loss, verts, faces = autoEnc(input_data, return_recon=True)
        # reconstructed_mesh = trimesh.Trimesh(verts.detach().numpy(), faces.detach().numpy())
        # reconstructed_mesh.show()

        for i in range(500):
            current_time = time.time()
            loss = autoEnc(input_data)
            print("loss: {}, time: {}, iter: {}".format(loss, time.time() - current_time, i))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # reconstruct the mesh
        loss, verts, faces = autoEnc(input_data, return_recon=True)
        reconstructed_mesh = trimesh.Trimesh(verts.cpu().numpy(), faces.cpu().numpy())
        reconstructed_mesh.show()
        reconstructed_mesh.export("reconstructed.obj")
    
