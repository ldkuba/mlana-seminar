import math
import torch
import torch.nn as nn
from x_transformers import Decoder
from x_transformers.autoregressive_wrapper import eval_decorator
from tqdm import tqdm

# taken from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class TokensPositionalEncoding(torch.nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 3000):
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
            x: Tensor, shape ``[seq_len, embedding_dim]``
        """
        pos = self.pe[:x.size(0)]
        x = x + pos

        return self.dropout(x)
    
def calculate_sliver_loss(vertices):
    # Compute the angle between the edges
    edges = vertices[:, (2, 0, 1)] - vertices
    edges = torch.nn.functional.normalize(edges, p=2, dim=-1)
    cos_angle_alpha = torch.sum(edges[:, 0] * -edges[:, 1], dim=-1)
    cos_angle_beta = torch.sum(edges[:, 1] * -edges[:, 2], dim=-1)
    cos_angle_gamma = torch.sum(edges[:, 2] * -edges[:, 0], dim=-1)
    cos_angle = torch.min(torch.cat((cos_angle_alpha, cos_angle_beta, cos_angle_gamma), dim=-1), dim=-1).values

    # raise to power to make the loss more sensitive to very small or very large angles
    cos_angle = torch.pow(cos_angle, 4)
    abs_cos_angle = torch.abs(cos_angle)

    # abs cos is in range [0, 1]
    sliver_loss_scale = 10.0
    return torch.sum(abs_cos_angle) * sliver_loss_scale

class MeshTransformer(torch.nn.Module):
    def __init__(self, autoencoder, token_dim=768, minimize_slivers=False):
        super().__init__()
        self.autoencoder = autoencoder
        self.token_dim = token_dim
        self.token_embedding = nn.Embedding(num_embeddings=autoencoder.codebook_size + 2, embedding_dim=token_dim)
        self.start_token = nn.Parameter(torch.randn(token_dim))
        self.end_token = nn.Parameter(torch.randn(token_dim))
        self.sos_id = autoencoder.codebook_size + 1
        self.eos_id = autoencoder.codebook_size

        self.context_length = 6 * 500
        self.positional_encoding = TokensPositionalEncoding(d_model=token_dim, max_len=self.context_length)

        self.decoder_only_transformer = Decoder(
            dim = self.token_dim,   # dimension of the model (token embeddings)
            depth = 24,             # depth of the network (layers)
            heads = 16,             # number of heads in the multiheadattention models
            attn_dim_head = 64,     # dimension of the heads
        )

        self.logits = nn.Linear(token_dim, autoencoder.codebook_size + 2)

    def freezeAutoEncoder(self):
        # freeze the autoencoder
        for param in self.autoencoder.parameters():
            param.requires_grad = False

    @eval_decorator
    @torch.no_grad()
    def generate(self, start_sequence, max_length=0):

        seq_length = max_length
        if max_length == 0:
            seq_length = self.context_length

        remaining_length = seq_length - start_sequence.size(0)
        codes = start_sequence

        cache = None

        for i in tqdm(range(remaining_length)):
            # Predict next token
            decoder_out, cache = self.process_codes(codes, cache=cache, append_eos=False, return_logits=True, return_cache=True)

            # Sample a random token from the distribution
            logits = decoder_out[-1, :]
            softmax_logits = torch.nn.functional.softmax(logits, dim=-1)

            # Can only predict eos on first token of a face
            if codes.size(0) % 6 > 0:
                softmax_logits[self.eos_id] = 0.0

            # Dont allow predicting sos token
            softmax_logits[self.sos_id] = 0.0

            next_token = torch.multinomial(softmax_logits, num_samples=1)

            if next_token == self.eos_id:
                break

            # Append the next token
            codes = torch.cat((codes, next_token))

        return codes

    def process_codes(self, face_codes, cache=None, append_eos=True, return_logits=False, return_cache=False, minimize_slivers=False):
        assert face_codes.size(0) <= self.context_length, "Input mesh is too large (max 500 faces)"

        # Save target face codes
        target = torch.cat((torch.tensor([self.sos_id], device='cuda'), face_codes.clone(), torch.tensor([self.eos_id], device='cuda')))

        # Create token embeddings
        face_codes = self.token_embedding(face_codes)

        # Add positional encoding
        face_codes = self.positional_encoding(face_codes)

        # Add sos and eos tokens
        face_codes = torch.cat((self.start_token.unsqueeze(0), face_codes))
        if append_eos:
            face_codes = torch.cat((face_codes, self.end_token.unsqueeze(0)))

        # Run the transformer decoder
        face_codes = face_codes.unsqueeze(0)

        if return_cache:
            face_decoded, cache = self.decoder_only_transformer(face_codes, cache=cache, return_hiddens=True)
        else:
            face_decoded = self.decoder_only_transformer(face_codes)
        
        face_decoded = face_decoded.squeeze(0)

        # Remove sos token and get logits
        face_logits = self.logits(face_decoded)
        
        if return_logits:
            if return_cache:
                return face_logits, cache
            else:
                return face_logits

        loss = torch.nn.functional.cross_entropy(face_logits, target)

        # Add sliver triangle penalty
        if minimize_slivers:
            # Remove sos and eos tokens
            face_logits = face_logits[1:-1, :]
            face_logits = torch.nn.functional.softmax(face_logits, dim=-1)
            face_logits[:, self.sos_id] = 0.0
            face_logits[:, self.eos_id] = 0.0
            face_codes = torch.multinomial(face_logits, num_samples=1)

            verts, faces = self.autoencoder.decode_mesh(face_codes.squeeze(1))
            face_vertices = verts[faces]
            loss += calculate_sliver_loss(face_vertices)

        if return_cache:
            return loss, cache
        else:
            return loss
        

    def forward(self, data, pad_value, minimize_slivers=False):
        # Encode mesh data
        with torch.no_grad():
            face_codes = self.autoencoder(data, pad_value, return_only_codes=True)
            # Flatten face tokens
            face_codes = face_codes.flatten(0)

        return self.process_codes(face_codes, append_eos=True, minimize_slivers=minimize_slivers)
