import math
import torch
import torch.nn as nn
import vector_quantize_pytorch as vq
import graph_encoder as ge
import decoder as dec
from torchvision.transforms import GaussianBlur
import scipy

# taken from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

def distribute_face_features_to_vertices(face_features, faces, vertex_count):
    vertex_features = torch.zeros(size=(vertex_count, 192))
    vertex_mean_count = torch.zeros(size=(vertex_count, 1))
    for i in range(faces.size(0)):
        vertex_features[faces[i][0]] = vertex_features[faces[i][0]] + face_features[i][:192]
        vertex_mean_count[faces[i][0]] += 1
        vertex_features[faces[i][1]] = vertex_features[faces[i][1]] + face_features[i][192:384]
        vertex_mean_count[faces[i][1]] += 1
        vertex_features[faces[i][2]] = vertex_features[faces[i][2]] + face_features[i][384:]
        vertex_mean_count[faces[i][2]] += 1

    vertex_features /= vertex_mean_count
    return vertex_features

def collect_vertex_features_to_faces(faces, quantized_vertex_features, vertex_codes):
    quantized_face_features = quantized_vertex_features[faces].flatten(1)
    face_codes = vertex_codes[faces].flatten(1)
    return quantized_face_features, face_codes

# vertices, angles, areas, normals are expected to be discretized
class AutoEncodeData():
    def __init__(self, face_vertices, vertex_count, faces, angles, face_areas, normals, edge_list):
        self.face_vertices = face_vertices
        self.vertex_count = vertex_count
        self.faces = faces
        self.angles = angles
        self.face_areas = face_areas
        self.normals = normals
        self.edge_list = edge_list

class AutoEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_discrete_values = 128
        # Values taken from https://github.com/lucidrains/meshgpt-pytorch/blob/main/meshgpt_pytorch/meshgpt_pytorch.py
        dim_vertex_embed = 64
        dim_angle_embed = 64
        dim_area_embed = 64
        dim_normal_embed = 64
        self.vertex_embedding = nn.Embedding(self.num_discrete_values, dim_vertex_embed)
        self.angles_embedding = nn.Embedding(self.num_discrete_values, dim_angle_embed)
        self.area_embedding = nn.Embedding(self.num_discrete_values, dim_area_embed)
        self.normal_embedding = nn.Embedding(self.num_discrete_values, dim_normal_embed)
        self.init_dim = dim_vertex_embed * 9 + dim_angle_embed * 3 + dim_normal_embed * 3 + dim_area_embed

        # Positional encoding for vertex coordinates
        self.positional_enc = PositionalEncoding(d_model=dim_vertex_embed)

        self.encoder = ge.GraphEncoder(self.init_dim)
        self.vector_quantizer = vq.ResidualVQ(dim=192, num_quantizers=2, codebook_size=16384, shared_codebook=True, stochastic_sample_codes = True, commitment_weight=1.0)
        self.decoder = dec.Decoder()

        gauss = scipy.signal.gaussian(5, 0.4)
        gauss = gauss / gauss.sum()
        self.smooth_kernel = torch.tensor(gauss).repeat(9).view(9,5).unsqueeze(1)
    
    def forward(self, data, return_recon=False):
        # embed face features
        vertex_embeddings = self.vertex_embedding(data.face_vertices)
        angles_embeddings = self.angles_embedding(data.angles)
        areas_embeddings = self.area_embedding(data.face_areas)
        normals_embeddings = self.normal_embedding(data.normals)

        # add positional embedding to vertex coordinates
        vertex_embeddings = self.positional_enc(vertex_embeddings)

        # concatenate all face features
        graph = torch.cat([vertex_embeddings, normals_embeddings, angles_embeddings, areas_embeddings[:, None]], dim=1).flatten(1)

        # Run the encoder
        encoded_graph = self.encoder(ge.GraphEncoderData(graph, data.edge_list))

        # Distribute face features across vertices
        vertex_features = distribute_face_features_to_vertices(encoded_graph, data.faces, data.vertex_count)

        # Run the vector quantizer on vertex features
        quantized_vertex_features, vertex_codes, commit_losses = self.vector_quantizer(vertex_features)

        # Collect the quantized indices back to faces
        quantized_face_features, face_codes = collect_vertex_features_to_faces(data.faces, quantized_vertex_features, vertex_codes)

        # Run the decoder
        decoded_vertices = self.decoder(quantized_face_features)
        num_decoded_faces = decoded_vertices.size(0) 
        
        # Split last dimension into coordinates per face
        decoded_vertices = decoded_vertices.view(num_decoded_faces, 9, -1)

        # Apply logmax
        decoded_vertices = torch.log_softmax(decoded_vertices, dim=-1)

        # Calculate target weights for loss function
        target_vertices_one_hot = torch.nn.functional.one_hot(data.face_vertices, num_classes=self.num_discrete_values).double()
        target_vertices_weights = torch.nn.functional.conv1d(target_vertices_one_hot, weight=self.smooth_kernel, groups=9, padding=2)

        # Calculate reconstruction loss
        reconstruction_loss = (-target_vertices_weights * decoded_vertices).sum()

        total_loss = reconstruction_loss + commit_losses.sum()

        if not return_recon:
            return total_loss
        
        with torch.no_grad():
            # Reconstruct the mesh from the decoded vertices
            # Get discretized vertex idx
            reconstructed_vertices_idx = decoded_vertices.argmax(dim=-1).double()

            # Get the actual coordinates and reshape to (num_faces, 3, 3)
            reconstructed_vertices = ((reconstructed_vertices_idx / (self.num_discrete_values - 1)) * 2 - 1)
            reconstructed_vertices = reconstructed_vertices.view(num_decoded_faces * 3, 3)

            reconstructed_faces = torch.arange(0, num_decoded_faces * 3).view(num_decoded_faces, 3)

            return total_loss, reconstructed_vertices, reconstructed_faces

