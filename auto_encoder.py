import math
import torch
import torch.nn as nn
import vector_quantize_pytorch as vq
import graph_encoder as ge

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

class Decoder(torch.nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, x):
        pass

class AutoEncodeData():
    def __init__(self, vertices, faces):
        self.vertices = vertices
        self.faces = faces

class AutoEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ge.GraphEncoder(16)
        self.vector_quantizer = vq.ResidualVQ(dim=192, num_quantizers=2, codebook_size=16384, shared_codebook=True, stochastic_sample_codes = True, commitment_weight=1.0)
        self.decoder = Decoder()
    
    def forward(self, data):
        # sort the mesh vertices and faces
        vertices_sorted, faces_sorted = ge.sort_mesh(data.vertices, data.faces)

        # compute angles, face areas and normals
        angles, face_areas, normals = ge.compute_angles_areas_normals(vertices_sorted, faces_sorted)

        # compute the graph, edge list
        graph = torch.cat([vertices_sorted[faces_sorted, :].flatten(1), normals, angles, face_areas[:, None]], dim=1)
        edge_list = ge.compute_edge_list(faces_sorted)
        graph_node_dims = graph.size(1)

        # Run the encoder
        encoded_graph = self.encoder(ge.GraphEncoderData(graph, edge_list))

        return encoded_graph
