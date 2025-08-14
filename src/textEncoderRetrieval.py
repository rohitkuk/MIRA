import torch
import torch.nn as nn
import numpy as np
from .positionalEmbedding import PositionalEmbedding
from .transformerEncoder import TransformerEncoder


class TextEncoderRetrieval(nn.Module):
    def __init__(self, vocab_size, d_model, max_seq_length, n_layers, n_heads, emb_dim):
        super().__init__()
        self.max_seq_length = max_seq_length
        self.embed = nn.Embedding(vocab_size, d_model)
        self.positional_embedding = PositionalEmbedding(d_model, max_seq_length)
        self.transformer_encoder = nn.ModuleList(
            [TransformerEncoder(d_model, n_heads) for _ in range(n_layers)]
        )
        self.projection = nn.Parameter(torch.randn(d_model, emb_dim))
        
    # For image retrieval
    def forward(self, text, mask=None):
        x = self.embed(text)
        x = self.positional_embedding(x)
    
        for encoder_layer in self.transformer_encoder:
            x = encoder_layer(x, mask=mask)
    
        if mask is not None:
            # Get the lengths of each sequence (i.e., find the last non-padded token)
            seq_lengths = mask.sum(dim=1) - 1  # Subtract 1 to get the index
            x = x[torch.arange(text.shape[0]), seq_lengths]
        else:
            x = x[:, -1]  # If no mask is provided, take the last token in the sequence.
    
        if self.projection is not None:
            x = x @ self.projection
    
        x = x / torch.norm(x, dim=-1, keepdim=True)
    
        return x