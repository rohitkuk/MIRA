
import torch
import torch.nn as nn
import numpy as np

class PositionalEmbedding(nn.Module):
        #------------------------------------------------------------------------------
        # PositionalEmbedding:
        # - Purpose: Inject positional information into token embeddings so that the 
        #   transformer can understand the order of tokens in a sequence.
        #
        # Args:
        #   d_model        → Embedding dimension of tokens (same as model input size)
        #   max_seq_length → Maximum sequence length for which we precompute positions
        #
        # Implementation (initialization step):
        # - `pe`: Tensor of shape [max_seq_length, d_model], initially all zeros.
        # - Will later be filled with sinusoidal values (fixed encoding) or learned 
        #   values depending on approach.
        # - This positional encoding will be added element-wise to token embeddings:
        #       X = TokenEmbedding + PositionalEmbedding
        #
        # Example shape:
        #   pe: [max_seq_length, d_model]
        #------------------------------------------------------------------------------

    def __init__(self, d_model, max_seq_length ):
        super().__init__()
        self.d_mdoel = d_model
        self.max_seq_length =max_seq_length
        
        pe = torch.zeros(max_seq_length, d_model)
        #------------------------------------------------------------------------------
        # Create position indices for each token in the sequence:
        # - `torch.arange(0, max_seq_length, dtype=torch.float)` → [0, 1, 2, ..., max_seq_length-1]
        #   Shape: [max_seq_length]
        # - `.unsqueeze(1)` → Add a dimension so shape becomes [max_seq_length, 1]
        #
        # Purpose:
        # - Each row now corresponds to a position index (0 to max_seq_length-1).
        # - This will later be combined with frequency terms to create sinusoidal encodings.
        #------------------------------------------------------------------------------
        position  = torch.arange(0, max_seq_length, dtype = torch.float).unsqueeze(1)
        
        # Sinusoidal = values follow sine/cosine wave patterns → used in Transformers so each sequence position 
        # gets a unique, continuous, and learnable-free encoding that preserves relative order. |

        div_term = torch.exp(
            torch.arange(0,d_model,2).float()* (-np.log(10000.0)/d_model)
        )
        # div_term: Frequency scaling factors for sinusoidal positional encoding.  
        # Purpose:
        #   In the Transformer formula:
        #       PE(pos, 2i)   = sin(pos / (10000^(2i/d_model)))
        #       PE(pos, 2i+1) = cos(pos / (10000^(2i/d_model)))
        #   This div_term precomputes (10000^(-2i/d_model)) for all even dimension indices.
        #
        # Steps:
        #   1. torch.arange(0, d_model, 2): select even indices (2i) for sine positions.
        #   2. Multiply by (-log(10000) / d_model): exponent term controlling frequency growth.
        #   3. torch.exp(...): convert from log space to real space → gives frequency denominators.
        #
        # Result:
        #   Each embedding dimension gets a unique wavelength, allowing positions to be encoded
        #   without training extra parameters. |

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        # ---------------------------------------------------------
        # Fill the positional encoding matrix (pe) with sine & cosine values:
        # - pe[:, 0::2] → Assigns sine values to even dimensions (0, 2, 4, ...)
        # - pe[:, 1::2] → Assigns cosine values to odd dimensions (1, 3, 5, ...)
        # Formula:
        #   PE[pos, 2i]   = sin(pos / (10000^(2i / d_model)))
        #   PE[pos, 2i+1] = cos(pos / (10000^(2i / d_model)))
        # Purpose:
        #   Sine and cosine allow the model to encode token positions
        #   in a way that preserves relative distances between tokens.
        #   (pos * div_term) ensures different frequencies per dimension.
        # register_buffer():
        #   Stores 'pe' in the module without treating it as a trainable parameter.
        #   pe.unsqueeze(0) → Adds batch dimension for broadcasting during forward pass.
        # ---------------------------------------------------------

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:,:seq_len]
    # ---------------------------------------------------------
    # Forward pass for positional encoding:
    # - seq_len = number of tokens in the current input sequence
    # - self.pe[:, :seq_len] → Slice the positional encodings to match seq_len
    # - Add positional encodings to token embeddings (x)
    # Purpose:
    #   This injects positional information into the embeddings,
    #   allowing the model to differentiate between tokens' positions.
    # Shapes:
    #   x                → [B, seq_len, d_model]
    #   self.pe[:, :L]   → [1, seq_len, d_model] (broadcasted over batch B)
    #   output           → [B, seq_len, d_model]
    # ---------------------------------------------------------

            
        
        

        