import math
import torch
import torch_geometric.nn as geom_nn
from collections import namedtuple

def inverse_permutation(perm):
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(perm.size(0), device=perm.device)
    return inv

def sort_mesh(vertices, faces):
    # sort vertices along x, y, z
    vertices_x_idx = torch.argsort(vertices[:, 0])
    vertices_x = vertices[vertices_x_idx]
    vertices_yx_idx = torch.argsort(vertices_x[:, 1], stable=True)
    vertices_yx = vertices_x[vertices_yx_idx]
    vertices_zyx_idx = torch.argsort(vertices_yx[:, 2], stable=True)
    vertices_sorted = vertices_yx[vertices_zyx_idx]
    # store a mapping from original vertex list to sorted vertex list
    vertices_original_idx_map = vertices_x_idx[vertices_yx_idx[vertices_zyx_idx]]
    vertices_original_idx_map_inv = inverse_permutation(vertices_original_idx_map)

    # replace the face indices with the sorted vertex indices
    faces = vertices_original_idx_map_inv[faces]

    # permute face indices to have lowest at the first index
    lowest_index_idx = torch.argmin(faces, dim=1)
    perms = (lowest_index_idx[:,None] + torch.arange(3).to(lowest_index_idx)) % 3
    faces = torch.gather(faces, 1, perms)

    # sort faces by lowest index of vertex
    faces_2_idx = torch.argsort(faces[:, 2])
    faces_2 = faces[faces_2_idx]
    faces_12_idx = torch.argsort(faces_2[:, 1], stable=True)
    faces_12 = faces_2[faces_12_idx]
    faces_012_idx = torch.argsort(faces_12[:, 0], stable=True)
    faces_sorted = faces_12[faces_012_idx]

    return (vertices_sorted, faces_sorted)

def compute_angles_areas_normals(vertices, faces):
    # compute face normals
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    e0 = v1 - v0
    e1 = v2 - v0
    e2 = v2 - v1
    face_normals = torch.cross(e0, e1)

    # compute face areas
    face_areas = 0.5 * torch.norm(face_normals, dim=1)

    # compute face angles
    e0_norm = torch.norm(e0, dim=1)
    e1_norm = torch.norm(e1, dim=1)
    e2_norm = torch.norm(e2, dim=1)

    face_angles = torch.stack([
        torch.acos(torch.clamp(torch.sum(e0 * e1, dim=1) / (e0_norm * e1_norm), -1.0, 1.0)),
        torch.acos(torch.clamp(torch.sum(-e0 * e2, dim=1) / (e0_norm * e2_norm), -1.0, 1.0)),
        torch.acos(torch.clamp(torch.sum(-e1 * -e2, dim=1) / (e1_norm * e2_norm), -1.0, 1.0))
    ], dim=1)

    return (face_angles, face_areas, face_normals)

# edge list between each pair of faces connected by an edge
def compute_edge_list(faces):
    vertex_pairs = torch.stack([faces[:, 0:2], faces[:, 1:3], faces[:, [2, 0]]], dim=1).sort(dim=-1).values

    edge_list = torch.empty(size=(0,2), dtype=torch.int64)

    for i in range(faces.size(0)-1):
        vp1 = (vertex_pairs[i+1:] == vertex_pairs[i][0]).all(dim=2).any(dim=1)
        vp2 = (vertex_pairs[i+1:] == vertex_pairs[i][1]).all(dim=2).any(dim=1)
        vp3 = (vertex_pairs[i+1:] == vertex_pairs[i][2]).all(dim=2).any(dim=1)
        adjecency = torch.logical_or(torch.logical_or(vp1, vp2), vp3)
        neighbours = torch.arange(start=i+1, end=faces.size(0))[adjecency]
        edge_list = torch.cat((edge_list, torch.stack((torch.ones(neighbours.size(0), dtype=torch.int)*i, neighbours), dim=1)))

    return edge_list

class GraphEncoderData():
    def __init__(self, x, edge_list):
        self.x = x
        self.edge_list = edge_list.long()

class GraphEncoder(torch.nn.Module):
    def __init__(self, input_dims):
        super().__init__()
        self.input_dims = input_dims
        self.conv1 = geom_nn.conv.SAGEConv(input_dims, 64, normalize=True, project=True)
        self.silu1 = torch.nn.SiLU()
        self.norm1 = torch.nn.LayerNorm(64)

        self.conv2 = geom_nn.conv.SAGEConv(64, 128, normalize=True, project=True)
        self.conv3 = geom_nn.conv.SAGEConv(128, 256, normalize=True, project=True)
        self.conv4 = geom_nn.conv.SAGEConv(256, 256, normalize=True, project=True)
        self.conv5 = geom_nn.conv.SAGEConv(256, 576, normalize=True, project=True)
    
    def forward(self, data) -> torch.Tensor:
        # SAGEConv expects the edge list in the shape (2, num_edges)
        data.edge_list = data.edge_list.transpose(0, 1)
        x = self.conv1(data.x, data.edge_list)
        x = self.conv2(x, data.edge_list)
        x = self.conv3(x, data.edge_list)
        x = self.conv4(x, data.edge_list)
        x = self.conv5(x, data.edge_list)
        return x
