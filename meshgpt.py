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
import wandb

import time

import trimesh

pad_value = -1
device = "cuda"


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
        trimesh_mesh = trimesh.load(filename, force="mesh")
        vertices = torch.tensor(
            trimesh_mesh.vertices, dtype=torch.float32, device="cpu"
        )
        faces = torch.tensor(trimesh_mesh.faces, dtype=torch.int64, device="cpu")

        # normalize the mesh to [-1, 1]
        aabb = trimesh_mesh.bounds.astype(np.float32)
        center = (aabb[0] + aabb[1]) / 2
        scale = max(aabb[1] - aabb[0])
        vertices = (vertices - center) / scale

        with torch.device(device):
            # sort the mesh vertices and faces
            vertices_sorted, faces_sorted = ge.sort_mesh(
                vertices.to(device), faces.to(device)
            )
            # compute edge list
            edge_list = ge.compute_edge_list(faces_sorted)

            # compute angles, face areas and normals
            angles, face_areas, normals = ge.compute_angles_areas_normals(
                vertices_sorted, faces_sorted
            )

            # distribute the vertices to the faces
            face_vertices = vertices_sorted[faces_sorted, :].flatten(1)

            return {
                "vertices": vertices_sorted.to("cpu"),
                "faces": faces_sorted.to("cpu"),
                "face_vertices": face_vertices.to("cpu"),
                "edge_list": edge_list.to("cpu"),
                "angles": angles.to("cpu"),
                "face_areas": face_areas.to("cpu"),
                "normals": normals.to("cpu"),
            }


def mesh_collate(data):
    values = [list(d.values()) for d in data]

    vertices, faces, face_vertices, edge_list, angles, face_areas, normals = zip(
        *values
    )

    vertices = torch.nn.utils.rnn.pad_sequence(
        vertices, batch_first=True, padding_value=pad_value
    )
    faces = torch.nn.utils.rnn.pad_sequence(
        faces, batch_first=True, padding_value=pad_value
    )
    face_vertices = torch.nn.utils.rnn.pad_sequence(
        face_vertices, batch_first=True, padding_value=0
    )
    edge_list = torch.nn.utils.rnn.pad_sequence(
        edge_list, batch_first=True, padding_value=0
    )
    angles = torch.nn.utils.rnn.pad_sequence(angles, batch_first=True, padding_value=0)
    face_areas = torch.nn.utils.rnn.pad_sequence(
        face_areas, batch_first=True, padding_value=0
    )
    normals = torch.nn.utils.rnn.pad_sequence(
        normals, batch_first=True, padding_value=0
    )

    # create face mask
    face_mask = (faces != pad_value).any(dim=-1)
    edge_mask = (edge_list != pad_value).any(dim=-1)

    return {
        "vertices": vertices,
        "faces": faces,
        "face_vertices": face_vertices,
        "edge_list": edge_list,
        "angles": angles,
        "face_areas": face_areas,
        "normals": normals,
        "face_mask": face_mask,
        "edeg_mask": edge_mask,
    }


class MeshGPTTrainer:
    def __init__(self):
        self.autoEnc = ae.AutoEncoder().to(device)
        self.autoenc_lr = 1e-4
        self.autoenc_batch_size = 64

        self.meshTransformer = mt.MeshTransformer(self.autoEnc, token_dim=512).to(
            device
        )
        self.transformer_lr = 1e-3
        self.transformer_batch_size = 32

    def train_autoencoder(
        self,
        dataset: MeshDataset,
        autoenc_dict_file=None, # File to save the trained autoencoder
        optimizer_dict_file=None, # If continuing training, supply the optimizer state dict
        save_every=-1,
        epochs=10,
        batch_size=None,
        lr=None,
        commit_weight=1.0,
        wandb_path=None,
        wandb_resume=False,
    ):

        # DONT FORGET TO SET MODEL TO TRAIN MODE
        self.autoEnc.train()

        # Override training params if requested
        if not batch_size:
            batch_size = self.autoenc_batch_size
        if not lr:
            lr = self.autoenc_lr

        # Initialize wandb
        if wandb_path:
            # 0 - entity, 1 - project, 2 - run
            wandb_path_split = wandb_path.split("/")
            if wandb_resume:
                # Find the run
                wandb_runs = wandb.Api().runs(
                    wandb_path_split[0] + "/" + wandb_path_split[1],
                    filters={"display_name": wandb_path_split[2]},
                )
                if len(wandb_runs) != 1:
                    raise ValueError("Invalid wandb path")

                # Resume it
                wandb.init(
                    entity=wandb_path_split[0],
                    project=wandb_path_split[1],
                    id=wandb_runs[0].id,
                    resume="must",
                    config={
                        "learning_rate": lr,
                        "architecture": "MeshGPT-Autoencoder",
                        "epochs": epochs,
                        "batch_size": batch_size,
                        "commit_weight": commit_weight,
                        "gauss_sigma": self.autoEnc.gauss_sigma,
                        "gauss_kernel_size": self.autoEnc.gauss_size,
                    },
                )
            else:
                wandb.init(
                    entity=wandb_path_split[0],
                    project=wandb_path_split[1],
                    name=wandb_path_split[2],
                    config={
                        "learning_rate": lr,
                        "architecture": "MeshGPT-Autoencoder",
                        "epochs": epochs,
                        "batch_size": batch_size,
                        "commit_weight": commit_weight,
                        "gauss_sigma": self.autoEnc.gauss_sigma,
                        "gauss_kernel_size": self.autoEnc.gauss_size,
                    },
                )
            wandb.watch(self.autoEnc, log="all", log_freq=save_every)

        num_batches = int(len(dataset) / batch_size)
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            generator=torch.Generator(device=device),
            collate_fn=mesh_collate,
        )

        with torch.device(device):

            autoencoderOptimizer = torch.optim.Adam(self.autoEnc.parameters(), lr=lr)
            if optimizer_dict_file:
                autoencoderOptimizer.load_state_dict(torch.load(optimizer_dict_file))

            for epoch in range(epochs):
                current_time = time.time()
                loss_avg = 0
                recon_loss_avg = 0
                commit_loss_avg = 0

                for batch_id, data in enumerate(data_loader):
                    for key in data:
                        data[key] = data[key].to(device)
                    loss, recon_loss, commit_loss = self.autoEnc(
                        data, return_detailed_loss=True, commit_weight=commit_weight
                    )
                    for key in data:
                        data[key] = data[key].to("cpu")
                    torch.cuda.empty_cache()

                    loss.backward()

                    loss_avg += loss.item()
                    recon_loss_avg += recon_loss.item()
                    commit_loss_avg += commit_loss.item()

                    autoencoderOptimizer.step()
                    autoencoderOptimizer.zero_grad()

                loss_avg /= num_batches
                recon_loss_avg /= num_batches
                commit_loss_avg /= num_batches
                print(
                    "loss: {}, recon_loss: {}, commit_loss: {}, time: {}, epoch: {}".format(
                        loss_avg,
                        recon_loss_avg,
                        commit_loss_avg,
                        time.time() - current_time,
                        epoch,
                    )
                )
                if wandb_path:
                    wandb.log(
                        {
                            "loss": loss_avg,
                            "recon_loss": recon_loss_avg,
                            "commit_loss": commit_loss_avg,
                        }
                    )

                if save_every > 0 and autoenc_dict_file:
                    if epoch % save_every == 0:
                        torch.save(
                            self.autoEnc.state_dict(),
                            autoenc_dict_file + "_epoch{}".format(epoch) + ".pth",
                        )
                        torch.save(
                            autoencoderOptimizer.state_dict(),
                            autoenc_dict_file
                            + "_optimizer_epoch{}".format(epoch)
                            + ".pth",
                        )

            # save the trained autoencoder
            if autoenc_dict_file:
                torch.save(self.autoEnc.state_dict(), autoenc_dict_file + "_end.pth")
                torch.save(
                    autoencoderOptimizer.state_dict(),
                    autoenc_dict_file + "_optimizer_end.pth",
                )

            if wandb_path:
                wandb.finish()

            # del data_loader
            del autoencoderOptimizer
            torch.cuda.empty_cache()

    def load_autoencoder(self, autoenc_dict_file):
        self.autoEnc.load_state_dict(torch.load(autoenc_dict_file))

    def reconstruct_mesh(self, in_mesh_files, wandb_name=None):

        # DONT FORGET TO SET MODEL TO EVAL MODE
        self.autoEnc.eval()

        if wandb_name:
            wandb.init(project="meshgpt-autoencoder", name=wandb_name)
            wandb.watch(self.autoEnc, log="all", log_freq=1)

        # reconstruct the mesh
        rec_dataset = MeshDataset(in_mesh_files)
        meshes = mesh_collate(rec_dataset.meshes)
        for key in meshes:
            meshes[key] = meshes[key].to(device)
        with torch.no_grad():
            verts, faces, loss, recon_loss, commit_loss = self.autoEnc(
                meshes, return_recon=True, return_detailed_loss=True
            )

        if wandb_name:
            wandb.log(
                {"loss": loss, "recon_loss": recon_loss, "commit_loss": commit_loss}
            )
            wandb.finish()

        return trimesh.Trimesh(verts[0].cpu().numpy(), faces[0].cpu().numpy())

    def train_mesh_transformer(
        self,
        dataset: MeshDataset,
        transformer_dict_file=None,
        optimizer_dict_file=None,
        save_every=-1,
        epochs=10,
        batch_size=None,
        lr=None,
        wandb_path=None,
        wandb_resume=False,
        minimize_slivers=True,
    ):
        # Set model to train mode
        self.meshTransformer.train()

        # Override training params if requested
        if not batch_size:
            batch_size = self.transformer_batch_size
        if not lr:
            lr = self.transformer_lr

        # TODO: Support batched training
        if batch_size > 1:
            raise ValueError("Batched training not supported")

        # Initialize wandb
        if wandb_path:
            # 0 - entity, 1 - project, 2 - run
            wandb_path_split = wandb_path.split("/")
            if wandb_resume:
                # Find the run
                wandb_runs = wandb.Api().runs(
                    wandb_path_split[0] + "/" + wandb_path_split[1],
                    filters={"display_name": wandb_path_split[2]},
                )
                if len(wandb_runs) != 1:
                    raise ValueError("Invalid wandb path")

                # Resume it
                wandb.init(
                    entity=wandb_path_split[0],
                    project=wandb_path_split[1],
                    id=wandb_runs[0].id,
                    resume="must",
                    config={
                        "learning_rate": lr,
                        "architecture": "MeshGPT-Transformer",
                        "epochs": epochs,
                        "batch_size": batch_size,
                        "sliver_minimization": minimize_slivers,
                    },
                )
            else:
                wandb.init(
                    entity=wandb_path_split[0],
                    project=wandb_path_split[1],
                    name=wandb_path_split[2],
                    config={
                        "learning_rate": lr,
                        "architecture": "MeshGPT-Transformer",
                        "epochs": epochs,
                        "batch_size": batch_size,
                        "sliver_minimization": minimize_slivers,
                    },
                )
            wandb.watch(self.meshTransformer, log="all", log_freq=save_every)

        num_batches = int(len(dataset) / batch_size)
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            generator=torch.Generator(device=device),
            collate_fn=mesh_collate,
        )

        with torch.device(device):

            # Freeze the autoencoder updates
            self.meshTransformer.freezeAutoEncoder()

            transformerOptimizer = torch.optim.Adam(
                self.meshTransformer.parameters(), lr=lr
            )
            if optimizer_dict_file:
                transformerOptimizer.load_state_dict(torch.load(optimizer_dict_file))

            # torch.cuda.memory._dump_snapshot("init_transformer.pickle")

            for epoch in range(epochs):
                current_time = time.time()
                loss_avg = 0

                for batch_id, data in enumerate(data_loader):
                    for key in data:
                        data[key] = data[key].to(device)
                    loss = self.meshTransformer(
                        data, pad_value, minimize_slivers=minimize_slivers
                    )
                    for key in data:
                        data[key] = data[key].to("cpu")
                    torch.cuda.empty_cache()

                    loss.backward()

                    loss_avg += loss.item()

                    transformerOptimizer.step()
                    transformerOptimizer.zero_grad()

                loss_avg /= num_batches
                print(
                    "loss: {}, time: {}, epoch: {}".format(
                        loss_avg,
                        time.time() - current_time,
                        epoch,
                    )
                )
                if wandb_path:
                    wandb.log({"loss": loss_avg})

                if save_every > 0 and transformer_dict_file:
                    if epoch % save_every == 0:
                        torch.save(
                            self.meshTransformer.state_dict(),
                            transformer_dict_file + "_epoch{}".format(epoch) + ".pth",
                        )
                        torch.save(
                            transformerOptimizer.state_dict(),
                            transformer_dict_file
                            + "_optimizer_epoch{}".format(epoch)
                            + ".pth",
                        )

            # Save the trained mesh transformer
            if transformer_dict_file:
                torch.save(
                    self.meshTransformer.state_dict(),
                    transformer_dict_file + "_end.pth",
                )
                torch.save(
                    transformerOptimizer.state_dict(),
                    transformer_dict_file + "_optimizer_end.pth",
                )

            if wandb_path:
                wandb.finish()

            del transformerOptimizer
            torch.cuda.empty_cache()

    def load_mesh_transformer(self, transformer_dict_file):
        self.meshTransformer.load_state_dict(torch.load(transformer_dict_file))

    def generate_mesh(self, prompt=None, max_length=0):

        # Dont forget to set model to eval mode
        self.meshTransformer.eval()

        if not prompt:
            prompt = torch.empty((0), device=device).long()

        # Generate a new mesh
        generated_codes = self.meshTransformer.generate(prompt, max_length)
        gen_verts, gen_faces = self.autoEnc.decode_mesh(generated_codes.unsqueeze(0))
        return trimesh.Trimesh(gen_verts[0].cpu().numpy(), gen_faces[0].cpu().numpy())
