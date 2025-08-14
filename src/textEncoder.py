import torch
import torch.nn as nn
from .positionalEmbedding import PositionalEmbedding
from .transformerEncoder import TransformerEncoder

class TextEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, max_seq_length, n_layers,n_heads, emb_dim):
        """
        Transformer-based text encoding module.

        This class encodes tokenized text sequences into dense vector embeddings
        using token embeddings, positional encodings, and multiple Transformer
        encoder layers, followed by a projection into a lower-dimensional space.

        Args:
            vocab_size (int): Number of unique tokens in the vocabulary.
            d_model (int): Dimensionality of token embeddings and hidden states.
            max_seq_length (int): Maximum input sequence length.
            n_layers (int): Number of Transformer encoder layers.
            n_heads (int): Number of attention heads per encoder layer.
            emb_dim (int): Output embedding dimension after projection.

        Attributes:
            embed (nn.Embedding): Embedding layer mapping tokens to dense vectors.
            positional_embeddings (PositionalEmbedding): Adds positional info to tokens.
            transformer_encoder (nn.ModuleList): Stack of Transformer encoder blocks.
            projecction (torch.nn.Parameter): Learnable projection from d_model to emb_dim.
        """
        super().__init__()
        self.max_seq_length = max_seq_length
        self.embed = nn.Embedding(vocab_size, d_model)
        self.positional_embeddings= PositionalEmbedding(d_model,max_seq_length)
        self.transformer_encoder = nn.ModuleList(
            [TransformerEncoder(d_model, n_heads) for _ in range(n_layers)]
        )
        self.projection = nn.Parameter(torch.randn(d_model, emb_dim))
        
    def forward(self, text, mask = None):
        x = self.embed(text)
        x = self.positional_embeddings(x)
        # iterativing calling each layer and give previous layer output as input to next iteration
        for encoder_layer in self.transformer_encoder:
            x = encoder_layer(x,mask = mask)
        # ------------------------------------------------------------------------------ |
        # This selects the last real (non-padding) token embedding (EOT) for each        |
        # sequence in the batch.                                                         |
        #                                                                                |
        # Visual:                                                                        |
        # x (encoder output) as a grid:                                                  |
        # Sentence 0 → [11]  [12]  [13]  [14]  [15]  [16]                                |
        # Sentence 1 → [21]  [22]  [23]  [24]  [25]  [26]                                |
        # Sentence 2 → [31]  [32]  [33]  [34]  [35]  [36]                                |
        #                                                                                |
        # mask (1 = real token, 0 = padding):                                            |
        # Sentence 0 →  1     1     1     0     0     0                                  |
        # Sentence 1 →  1     1     1     1     1     0                                  |
        # Sentence 2 →  1     1     0     0     0     0                                  |
        #                                                                                |
        # Step-by-step:                                                                  |
        # 1. Count 1s in mask → sequence lengths: [3, 5, 2]                              |
        # 2. Subtract 1 → last token indices: [2, 4, 1]                                  |
        # 3. Advanced indexing:                                                          |
        #    Row indices:    [0, 1, 2]                                                   |
        #    Column indices: [2, 4, 1]                                                   |
        # 4. Picks: [13], [25], [32]  → the EOT embeddings.                              |
        # ------------------------------------------------------------------------------ |
        x = x[torch.arange(text.shape[0]), torch.sub(torch.sum(mask[:,0], dim=1),1)]
        
        if self.projection is not None:
            x = x @ self.projection
            
        # -------------------------------------------------------------------
        # Step: Normalize embeddings to unit length
        #
        # Purpose:
        #   - Compute the L2 norm (vector magnitude) for each embedding 
        #     along the last dimension using `torch.norm`.
        #   - Divide each embedding vector by its own norm to scale its 
        #     length (magnitude) to exactly 1.
        #   - This keeps only the *direction* of the embedding vector 
        #     relevant, making it useful for cosine similarity and 
        #     comparison tasks.
        #
        # Details:
        #   - `dim=-1` → Operates along the last axis (embedding dimension).
        #   - `keepdim=True` → Retains shape for broadcasting during division.
        #
        # Example:
        #   Input:  x = [[3,4], [1,2]]
        #           Norms = [5, √5] ≈ [5.0, 2.236]
        #           Output = [[0.6, 0.8], [0.447, 0.894]]
        # -------------------------------------------------------------------
        x = x/torch.norm(x, dim = -1, keepdim = True)
        return x
