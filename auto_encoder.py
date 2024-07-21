import math
import torch
import torch.nn as nn
import vector_quantize_pytorch as vq
import graph_encoder as ge
import decoder as dec
import numpy as np

# taken from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 20000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        num_faces = x.size(-3)

        pos = self.pe[:x.size(-3)].unsqueeze(-2)
        pos = pos.expand(num_faces, 9, self.d_model)
        x = x + pos

        return self.dropout(x)

def distribute_face_features_to_vertices(face_features, faces, vertex_count):
    batch_size = face_features.size(0)
    num_faces = face_features.size(1)
    
    # Reshape face features to (batch, num_faces, 3, vertex_dim)
    face_features = face_features.reshape(batch_size, num_faces, 3, -1)
    vertex_dim = face_features.size(-1)

    # Initialize vertex features
    vertex_features = torch.zeros((batch_size, vertex_count, vertex_dim), device=face_features.device)

    # Add additional vertex id to account for variable length vertex lists
    vertex_features = torch.nn.functional.pad(vertex_features, (0, 0, 0, 1), value=0.0)
    vertex_features_mask = torch.ones((batch_size, vertex_count+1), device=face_features.device, dtype=torch.bool)
    vertex_features_mask[:, -1] = False

    faces_with_dim = faces.flatten(1).repeat_interleave(vertex_dim, dim=-1).reshape(batch_size, -1, vertex_dim)
    face_features = face_features.reshape(batch_size, -1, vertex_dim)

    # Scatter mean
    num = vertex_features.scatter_add(dim=-2, index=faces_with_dim, src=face_features)
    den = torch.zeros_like(vertex_features).scatter_add(dim=-2, index=faces_with_dim, src=torch.ones_like(face_features))
    avg_vertex_features = num / den.clamp(min=1e-5)

    return avg_vertex_features, vertex_features_mask


def collect_vertex_features_to_faces(faces, quantized_vertex_features, vertex_codes):
    faces_flattened = faces.flatten(1).unsqueeze(-1)
    quantized_face_features = torch.gather(quantized_vertex_features, 1, faces_flattened.expand(-1, -1, quantized_vertex_features.size(-1))).reshape(faces.size(0), faces.size(1), -1)
    face_codes = torch.gather(vertex_codes, 1, faces_flattened.expand(-1, -1, vertex_codes.size(-1))).reshape(faces.size(0), faces.size(1), -1)
    return quantized_face_features, face_codes

def discretize(x, min_val, max_val, num_values):
    x = (x - min_val) / (max_val - min_val)
    x = x * (num_values - 1)
    x = torch.round(x).long()
    return x

def gaussian_filter1d(size,sigma):
    filter_range = np.linspace(-int(size/2),int(size/2),size)
    gaussian_filter = [1 / (sigma * np.sqrt(2*np.pi)) * np.exp(-x**2/(2*sigma**2)) for x in filter_range]
    return gaussian_filter

class AutoEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_discrete_values = 128
        # Values taken from https://github.com/lucidrains/meshgpt-pytorch/blob/main/meshgpt_pytorch/meshgpt_pytorch.py
        dim_vertex_embed = 64
        dim_angle_embed = 16
        dim_area_embed = 16
        dim_normal_embed = 64
        self.vertex_embedding = nn.Embedding(self.num_discrete_values, dim_vertex_embed)
        self.angles_embedding = nn.Embedding(self.num_discrete_values, dim_angle_embed)
        self.area_embedding = nn.Embedding(self.num_discrete_values, dim_area_embed)
        self.normal_embedding = nn.Embedding(self.num_discrete_values, dim_normal_embed)
        self.init_dim = dim_vertex_embed * 9 + dim_angle_embed * 3 + dim_normal_embed * 3 + dim_area_embed

        self.face_node_dim = 196
        self.codebook_size = 16384

        self.linear_face_node = nn.Linear(self.init_dim, self.face_node_dim)

        # Positional encoding for vertex coordinates
        self.positional_enc = PositionalEncoding(d_model=dim_vertex_embed)

        self.encoder = ge.GraphEncoder(self.face_node_dim)
        self.vector_quantizer = vq.ResidualVQ(dim=192, num_quantizers=2, codebook_size=self.codebook_size, shared_codebook=True, stochastic_sample_codes = True, commitment_weight=1.0)
        self.decoder = dec.Decoder()

        self.gauss_size = 5
        gauss = torch.tensor(gaussian_filter1d(self.gauss_size, 0.1), device='cuda')
        self.smooth_kernel = gauss / gauss.sum()

    @torch.no_grad()
    def decode_mesh(self, face_codes):
        batch_size = face_codes.size(0)

        # Get quantized vectors from codebooks
        face_codes = face_codes.view(batch_size, -1, 2)
        quantized = self.vector_quantizer.get_output_from_indices(face_codes)

        # Run the decoder
        decoded_vertices = self.decoder(quantized)

        num_decoded_faces = decoded_vertices.size(1)

        # Split last dimension into coordinates per face
        decoded_vertices = decoded_vertices.view(batch_size, num_decoded_faces, 9, -1)

        # Apply logmax
        decoded_vertices = torch.log_softmax(decoded_vertices, dim=-1)

        return self.decoded_vertices_to_mesh(decoded_vertices)
    
    @torch.no_grad()
    def decoded_vertices_to_mesh(self, decoded_vertices):
        batch_size = decoded_vertices.size(0)
        num_decoded_faces = decoded_vertices.size(1)

        # Reconstruct the mesh from the decoded vertices
        # Get discretized vertex idx
        reconstructed_vertices_idx = decoded_vertices.argmax(dim=-1).double()

        # Get the actual coordinates and reshape to (batch_size, num_faces, 3, 3)
        reconstructed_vertices = ((reconstructed_vertices_idx / (self.num_discrete_values - 1)) * 2 - 1)
        reconstructed_vertices = reconstructed_vertices.view(batch_size, num_decoded_faces * 3, 3)

        reconstructed_faces = torch.arange(0, num_decoded_faces * 3).view(num_decoded_faces, 3).unsqueeze(0).expand(batch_size, -1, -1)

        return reconstructed_vertices, reconstructed_faces


    def reconstruction_loss(self, decoded_vertices, face_mask, batch_size, vertex_discrete):
        # Reshape to (batch, discrete_values, ...)
        decoded_vertices = decoded_vertices.view(batch_size, -1, self.num_discrete_values).swapaxes(1, 2)
        
        # Apply logmax
        decoded_vertices = torch.log_softmax(decoded_vertices, dim=1)

        # Calculate target weights for loss function
        target_vertices_one_hot = torch.nn.functional.one_hot(vertex_discrete, num_classes=self.num_discrete_values).double()
        target_vertices_one_hot = target_vertices_one_hot.view(batch_size, -1, self.num_discrete_values)
        groups = target_vertices_one_hot.size(-2)

        smooth_kernel = self.smooth_kernel.repeat(groups).view(groups, self.gauss_size).unsqueeze(1)
        target_vertices_weights = torch.nn.functional.conv1d(target_vertices_one_hot, weight=smooth_kernel, groups=groups, padding=self.gauss_size//2)
        target_vertices_weights = target_vertices_weights.swapaxes(1, 2)

        # Calculate reconstruction loss
        reconstruction_loss = (-target_vertices_weights * decoded_vertices).sum(dim=1)
        recon_face_mask = face_mask.repeat_interleave(9, dim=1)
        reconstruction_loss = reconstruction_loss[recon_face_mask].mean()

        return reconstruction_loss

    # Data:
    # - vertices: Tensor of shape (batch, num_vertices, 3)
    # - faces: Tensor of shape (batch, num_faces, 3)
    # - edge_list: Tensor of shape (batch, num_edges, 2)6
    # - angles: Tensor of shape (batch, num_faces, 3)
    # - face_areas: Tensor of shape (batch, num_faces)
    # - normals: Tensor of shape (batch, num_faces, 3)
    # === MASKS ===
    # - face_mask: Tensor of shape (batch, num_faces, 3)
    # - edge_mask: Tensor of shape (batch, num_edges, 2)
    def forward(self, data, return_recon=False, return_only_codes=False, return_detailed_loss=False, commit_weight=1.0):
        faces = data['faces']
        vertices = data['vertices']
        num_vertices = vertices.size(-2)
        num_faces = faces.size(-2)
        batch_size = faces.size(0)

        # quantize face features
        vertex_discrete = discretize(data['face_vertices'] , -1, 1, self.num_discrete_values)
        angles_discrete = discretize(data['angles'], 0, math.pi, self.num_discrete_values)
        areas_discrete = discretize(data['face_areas'], 0, 4, self.num_discrete_values)
        normals_discrete = discretize(data['normals'], -1, 1, self.num_discrete_values)

        # embed face features
        vertex_embeddings = self.vertex_embedding(vertex_discrete)
        angles_embeddings = self.angles_embedding(angles_discrete)
        areas_embeddings = self.area_embedding(areas_discrete)
        normals_embeddings = self.normal_embedding(normals_discrete)

        # add positional embedding to vertex coordinates
        vertex_embeddings = self.positional_enc(vertex_embeddings)

        # concatenate all face features
        graph = torch.cat([vertex_embeddings.flatten(-2), normals_embeddings.flatten(-2), angles_embeddings.flatten(-2), areas_embeddings], dim=-1)
        graph = self.linear_face_node(graph)

        # Remove masked faces from graph and concatenate edge list for GraphEncoder
        graph_masked = graph[data['face_mask']]
        index_offsets = torch.cumsum(torch.sum(data['face_mask'], dim=-1), dim=-1)
        index_offsets = torch.cat([torch.zeros(1, device=index_offsets.device), index_offsets], dim=0)[:-1]

        offset_edge_list = data['edge_list'] + index_offsets.unsqueeze(-1).unsqueeze(-1)
        offset_edge_list = offset_edge_list.flatten(0, 1)

        # Run the encoder
        encoded_graph = self.encoder(ge.GraphEncoderData(graph_masked, offset_edge_list))

        # Reshape the graph back to batches
        original_shape = (*graph.shape[:2], encoded_graph.size(-1))
        encoded_graph = encoded_graph.new_zeros(original_shape).masked_scatter(data['face_mask'].unsqueeze(-1), encoded_graph)

        # Distribute face features across vertices
        pad_vertex_id = num_vertices
        faces_max_vertex_padding = faces.masked_fill(~data['face_mask'].unsqueeze(-1), pad_vertex_id)
        vertex_features, vertex_features_mask = distribute_face_features_to_vertices(encoded_graph, faces_max_vertex_padding, num_vertices)

        # Run the vector quantizer on vertex features
        quantized_vertex_features, vertex_codes, commit_losses = self.vector_quantizer(vertex_features, mask=vertex_features_mask)

        # Collect the quantized indices back to faces
        quantized_face_features, face_codes = collect_vertex_features_to_faces(faces_max_vertex_padding, quantized_vertex_features, vertex_codes)

        # terminate early if we just perfom the encoding + quantization step
        if return_only_codes:
            return face_codes

        # Run the decoder
        decoder_face_mask = data['face_mask'].unsqueeze(1)
        decoded_vertices = self.decoder(quantized_face_features, mask=decoder_face_mask)

        if return_recon:
            with torch.no_grad():
                # Split last dimension into coordinates per face
                decoded_vertices = decoded_vertices.view(batch_size, num_faces, 9, self.num_discrete_values)

                reconstructed_vertices, reconstructed_faces = self.decoded_vertices_to_mesh(decoded_vertices)

                if return_detailed_loss:
                    recon_loss = self.reconstruction_loss(decoded_vertices, data['face_mask'], batch_size, vertex_discrete)
                    commit_loss = commit_losses.sum() * commit_weight
                    total_loss = recon_loss + commit_loss
                    return reconstructed_vertices, reconstructed_faces, total_loss, recon_loss, commit_loss

                return reconstructed_vertices, reconstructed_faces

        recon_loss = self.reconstruction_loss(decoded_vertices, data['face_mask'], batch_size, vertex_discrete)

        commit_loss = commit_losses.sum() * commit_weight
        total_loss = recon_loss + commit_loss

        if return_detailed_loss:
            return total_loss, recon_loss, commit_loss
        
        return total_loss

