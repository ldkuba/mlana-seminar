import math
import torch
import numpy as np
from torch import nn, Tensor
import torch.utils
import trimesh.viewer
import auto_encoder as ae
import graph_encoder as ge
from torch.utils.data import Dataset, DataLoader

import time

import trimesh

num_discrete_values = 128
pad_value = -2

class MeshDataset(Dataset):
    def __init__(self, meshes):
        self.meshes = [load_model(x) for x in meshes]

    def __len__(self):
        return len(self.meshes)

    def __getitem__(self, idx):
        return self.meshes[idx]

# Loads the model and returns the sorted vertices, faces and edge list
def load_model(filename):
    # load the mesh
    trimesh_mesh = trimesh.load_mesh(filename, merge_tex=True, merge_norm=True)
    vertices = torch.tensor(trimesh_mesh.vertices, dtype=torch.float32)
    faces = torch.tensor(trimesh_mesh.faces, dtype=torch.int64)

    # normalize the mesh to [-1, 1]
    aabb = trimesh_mesh.bounds.astype(np.float32)
    center = (aabb[0] + aabb[1]) / 2
    scale = max(aabb[1] - aabb[0])
    vertices = (vertices - center) / scale

    with torch.device("cpu"):
        # sort the mesh vertices and faces
        vertices_sorted, faces_sorted = ge.sort_mesh(vertices.to('cpu'), faces.to('cpu'))
        # compute edge list
        edge_list = ge.compute_edge_list(faces_sorted)

        # compute angles, face areas and normals
        angles, face_areas, normals = ge.compute_angles_areas_normals(vertices_sorted, faces_sorted)

        # distribute the vertices to the faces
        face_vertices = vertices_sorted[faces_sorted, :].flatten(1)

        return {
            'vertices': vertices_sorted,
            'faces': faces_sorted,
            'face_vertices': face_vertices,
            'edge_list': edge_list,
            'angles': angles,
            'face_areas': face_areas,
            'normals': normals
        }
    
def mesh_collate(data):
    values = [list(d.values()) for d in data]

    vertices, faces, face_vertices, edge_list, angles, face_areas, normals = zip(*values)
    
    vertices = torch.nn.utils.rnn.pad_sequence(vertices, batch_first=True, padding_value=pad_value)
    faces = torch.nn.utils.rnn.pad_sequence(faces, batch_first=True, padding_value=pad_value)
    face_vertices = torch.nn.utils.rnn.pad_sequence(face_vertices, batch_first=True, padding_value=0)
    edge_list = torch.nn.utils.rnn.pad_sequence(edge_list, batch_first=True, padding_value=0)
    angles = torch.nn.utils.rnn.pad_sequence(angles, batch_first=True, padding_value=0)
    face_areas = torch.nn.utils.rnn.pad_sequence(face_areas, batch_first=True, padding_value=0)
    normals = torch.nn.utils.rnn.pad_sequence(normals, batch_first=True, padding_value=0)

    # create face mask
    face_mask = faces.any(dim=-1) != pad_value
    edge_mask = edge_list.any(dim=-1) != pad_value

    return {'vertices': vertices,
            'faces': faces,
            'face_vertices': face_vertices,
            'edge_list': edge_list,
            'angles': angles,
            'face_areas': face_areas,
            'normals': normals,
            'face_mask': face_mask,
            'edeg_mask': edge_mask}

if __name__ == "__main__":
    # parameters
    lr = 1e-1

    meshes = ["lantern.obj", "octopussy.obj"]

    dataset = MeshDataset(meshes)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True, drop_last=True, generator=torch.Generator(device='cpu'), collate_fn=mesh_collate)

    with torch.device("cpu"):
        autoEnc = ae.AutoEncoder().to("cpu")
        optimizer = torch.optim.Adam(autoEnc.parameters(), lr=lr)

        for batch_id, data in enumerate(data_loader):
            current_time = time.time()
            loss = autoEnc(data, pad_value)
            print("loss: {}, time: {}, batch: {}".format(loss, time.time() - current_time, batch_id))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # reconstruct the mesh
        rec_model = load_model("lantern.obj")
        loss, verts, faces = autoEnc(rec_model, pad_value, return_recon=True)
        reconstructed_mesh = trimesh.Trimesh(verts.cpu().numpy(), faces.cpu().numpy())
        reconstructed_mesh.show()
        reconstructed_mesh.export("reconstructed.obj")
    
