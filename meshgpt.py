import math
import torch
import numpy as np
from torch import nn, Tensor
import torch.utils
import trimesh.viewer
import auto_encoder as ae
import graph_encoder as ge
import mesh_transformer as mt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import time

import trimesh

pad_value = -2
device = 'cuda'

class MeshDataset(Dataset):
    def __init__(self, meshes=[]):
        self.meshes = []
        if len(meshes) == 0:
            return
        print("Loading and processing raw dataset")
        for x in tqdm(meshes):
            self.meshes.append(MeshDataset.load_model(x))

    def __len__(self):
        return len(self.meshes)

    def __getitem__(self, idx):
        return self.meshes[idx]
    
    def save(self, filename):
        torch.save(self.meshes, filename)

    def load(self, filename):
        print("Loading processed dataset")
        self.meshes = torch.load(filename)

    # Loads the model and returns the sorted vertices, faces and edge list
    def load_model(filename):
        # load the mesh
        trimesh_mesh = trimesh.load(filename, merge_tex=True, merge_norm=True, force='mesh')
        vertices = torch.tensor(trimesh_mesh.vertices, dtype=torch.float32, device='cpu')
        faces = torch.tensor(trimesh_mesh.faces, dtype=torch.int64, device='cpu')

        # normalize the mesh to [-1, 1]
        aabb = trimesh_mesh.bounds.astype(np.float32)
        center = (aabb[0] + aabb[1]) / 2
        scale = max(aabb[1] - aabb[0])
        vertices = (vertices - center) / scale

        with torch.device(device):
            # sort the mesh vertices and faces
            vertices_sorted, faces_sorted = ge.sort_mesh(vertices.to(device), faces.to(device))
            # compute edge list
            edge_list = ge.compute_edge_list(faces_sorted)

            # compute angles, face areas and normals
            angles, face_areas, normals = ge.compute_angles_areas_normals(vertices_sorted, faces_sorted)

            # distribute the vertices to the faces
            face_vertices = vertices_sorted[faces_sorted, :].flatten(1)

            return {
                'vertices': vertices_sorted.to('cpu'),
                'faces': faces_sorted.to('cpu'),
                'face_vertices': face_vertices.to('cpu'),
                'edge_list': edge_list.to('cpu'),
                'angles': angles.to('cpu'),
                'face_areas': face_areas.to('cpu'),
                'normals': normals.to('cpu')
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

class MeshGPTTrainer():
    def __init__(self, dataset):
        self.autoEnc = ae.AutoEncoder().to(device)
        self.autoenc_lr = 1e-4
        self.autoenc_batch_size = 1

        self.meshTransformer = mt.MeshTransformer(self.autoEnc, token_dim=512).to(device)
        self.transformer_lr = 1e-3
        self.transformer_batch_size = 32

        self.dataset = dataset

    def train_autoencoder(self, autoenc_dict_file=None, optimizer_dict_file=None, save_every=-1, epochs=10):
        num_batches = int(len(self.dataset) / self.autoenc_batch_size)
        # data_loader = DataLoader(self.dataset, batch_size=self.autoenc_batch_size, shuffle=True, drop_last=True, generator=torch.Generator(device=device), collate_fn=mesh_collate)

        with torch.device(device):
            
            autoencoderOptimizer = torch.optim.Adam(self.autoEnc.parameters(), lr=self.autoenc_lr)
            if optimizer_dict_file:
                autoencoderOptimizer.load_state_dict(torch.load(optimizer_dict_file))

            for epoch in range(epochs):
                # TODO: make model work properly with batch_size > 1. For now manual batches
                # for batch_id, data in enumerate(data_loader):
                for batch_id in range(num_batches):
                    current_time = time.time()

                    total_loss = 0
                    for i in range(self.autoenc_batch_size):
                        data = self.dataset[batch_id * self.autoenc_batch_size + i]
                        for key in data:
                            data[key] = data[key].to(device)
                        total_loss += self.autoEnc(data, pad_value)
                        for key in data:
                            data[key] = data[key].to('cpu')
                        torch.cuda.empty_cache()

                    total_loss /= self.autoenc_batch_size
                    total_loss.backward()

                    autoencoderOptimizer.step()
                    autoencoderOptimizer.zero_grad()
                    print("loss: {}, time: {}, batch: {}, epoch: {}".format(total_loss, time.time() - current_time, batch_id, epoch))

                    if save_every > 0 and autoenc_dict_file:
                        if batch_id == num_batches - 1:
                            torch.save(self.autoEnc.state_dict(), autoenc_dict_file + "_epoch{}".format(epoch) + "_last.pth")
                            torch.save(autoencoderOptimizer.state_dict(), autoenc_dict_file + "_epoch{}".format(epoch) + "_optimizer_last.pth")
                        elif batch_id % save_every == 0:
                            torch.save(self.autoEnc.state_dict(), autoenc_dict_file + "_epoch{}".format(epoch) + "_batch{}".format(batch_id) + ".pth")
                            torch.save(autoencoderOptimizer.state_dict(), autoenc_dict_file + "_epoch{}".format(epoch) + "_optimizer_batch{}".format(batch_id) + ".pth")

            # save the trained autoencoder
            if autoenc_dict_file:
                torch.save(self.autoEnc.state_dict(), autoenc_dict_file + "_end.pth")
                torch.save(autoencoderOptimizer.state_dict(), autoenc_dict_file + "_optimizer_end.pth")

            # del data_loader
            del autoencoderOptimizer
            torch.cuda.empty_cache()

    def load_autoencoder(self, autoenc_dict_file):
        self.autoEnc.load_state_dict(torch.load(autoenc_dict_file))

    def reconstruct_mesh(self, in_mesh_file):
        # reconstruct the mesh
        rec_model = MeshDataset.load_model(in_mesh_file)
        for key in rec_model:
            rec_model[key] = rec_model[key].to(device)
        loss, verts, faces = self.autoEnc(rec_model, pad_value, return_recon=True)
        return trimesh.Trimesh(verts.cpu().numpy(), faces.cpu().numpy())

    def train_mesh_transformer(self, transformer_dict_file=None, save_every=8, epochs=1, minimize_slivers=True):
        num_batches = int(len(self.dataset) / self.transformer_batch_size)
        # data_loader = DataLoader(self.dataset, batch_size=self.transformer_batch_size, shuffle=True, drop_last=True, generator=torch.Generator(device=device), collate_fn=mesh_collate)

        with torch.device(device):
            transformerOptimizer = torch.optim.Adam(self.meshTransformer.parameters(), lr=self.transformer_lr)
            self.meshTransformer.freezeAutoEncoder()

            torch.cuda.memory._dump_snapshot("init_transformer.pickle")

            for epoch in range(epochs):
                for batch_id in range(num_batches):
                    current_time = time.time()

                    total_loss = 0
                    for i in range(self.transformer_batch_size):
                        data = self.dataset[batch_id * self.transformer_batch_size + i]
                        for key in data:
                            data[key] = data[key].to(device)
                        total_loss += self.meshTransformer(data, pad_value, minimize_slivers=minimize_slivers)
                        for key in data:
                            data[key] = data[key].to('cpu')
                        torch.cuda.empty_cache()

                    total_loss /= self.transformer_batch_size
                    total_loss.backward()
                    
                    transformerOptimizer.step()
                    transformerOptimizer.zero_grad()
                    print("loss: {}, time: {}, batch: {}, epoch: {}".format(total_loss, time.time() - current_time, batch_id, epoch))

                    if save_every > 0 and transformer_dict_file:
                        if batch_id == num_batches - 1:
                            torch.save(self.autoEnc.state_dict(), transformer_dict_file + "_epoch{}".format(epoch) + "_last.pth")
                            torch.save(transformerOptimizer.state_dict(), transformer_dict_file + "_epoch{}".format(epoch) + "_optimizer_last.pth")
                        elif batch_id % save_every == 0:
                            torch.save(self.autoEnc.state_dict(), transformer_dict_file + "_epoch{}".format(epoch) + "_batch{}".format(batch_id) + ".pth")
                            torch.save(transformerOptimizer.state_dict(), transformer_dict_file + "_epoch{}".format(epoch) + "_optimizer_batch{}".format(batch_id) + ".pth")

            # Save the trained mesh transformer
            if transformer_dict_file:
                torch.save(self.meshTransformer.state_dict(), transformer_dict_file + "_end.pth")
                torch.save(transformerOptimizer.state_dict(), transformer_dict_file + "_optimizer_end.pth")

            del transformerOptimizer
            torch.cuda.empty_cache()

    def load_mesh_transformer(self, transformer_dict_file):
        self.meshTransformer.load_state_dict(torch.load(transformer_dict_file))

    def generate_mesh(self, prompt=None, max_length=0):
        if not prompt:
            prompt = torch.empty((0), device=device).long()

        # Generate a new mesh
        generated_codes = self.meshTransformer.generate(prompt, max_length)
        gen_verts, gen_faces = self.autoEnc.decode_mesh(generated_codes)
        return trimesh.Trimesh(gen_verts.cpu().numpy(), gen_faces.cpu().numpy())
