from .multiAttentionHead import MultiHeadAttention
import torch.nn as nn

class TransformerEncoder(nn.Module):
    
    def __init__(self, d_model, n_heads, mlp_ratio = 4):
        
        """
        
        Args:
        d_model (int):      Dimensionality of input & output embeddings (e.g. 512 or 768).
        n_heads (int):      Number of parallel attention heads (e.g. 8 or 12).
        mlp_ratio (float):  Multiplier for hidden size of the position-wise feed-forward layer.
                                If mlp_ratio=4 and d_model=512, then hidden dim = 512 * 4 = 2048.
                            
        """
        #The FFN (governed by mlp_ratio) injects non-linear transformations
        # so the model can learn complex functions beyond attention.
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.mlp_ratio = mlp_ratio
        
        # -----------------------------------------------------------------------|
        # Layer Normalization (Pre-Norm)                                         |
        # -----------------------------------------------------------------------|
        # Applies layer normalization to stabilize and speed up training.        |
        # It normalizes the input across the embedding dimension.                |
        #                                                                        |
        # Formula:                                                               |
        #   LN(x) = (x - mean(x)) / sqrt(var(x) + ε)                             |
        #                                                                        |
        # - Used before Multi-Head Attention to help gradient flow.              |
        # - Prevents internal covariate shift.                                   |
        # -----------------------------------------------------------------------|
        self.ln1 = nn.LayerNorm(d_model)
        
        # -----------------------------------------------------------------------|
        # Multi-Head Attention Layer                                             |
        # -----------------------------------------------------------------------|
        # Applies self-attention using multiple heads in parallel.               |
        # Each head learns different relationships between tokens.               |
        #                                                                        |
        # - Internally performs:                                                 |
        #     Q = xW_Q,  K = xW_K,  V = xW_V                                     |
        #     Attention = Softmax(QKᵗ / √dₖ) · V                                 |
        # - Then concatenates outputs of all heads and projects via W_o.         |
        #                                                                        |
        # Helps the model focus on information from different representation     |
        # subspaces and positions.                                               |
        # -----------------------------------------------------------------------|
        self.mha = MultiHeadAttention(d_model, n_heads)
        
        
        # -----------------------------------------------------------------------|
        # Layer Normalization 2                                                  |
        # -----------------------------------------------------------------------|
        # Applies normalization to the output of the multi-head attention +      |
        # residual connection before feeding it into the feed-forward network.   |
        # This helps stabilize training and maintain scale of representations.   |
        # -----------------------------------------------------------------------|
        self.ln2 = nn.LayerNorm(d_model)
        
        
        # -----------------------------------------------------------------------|
        # Feed Forward Network (MLP)                                             |
        # -----------------------------------------------------------------------|
        # A two-layer fully connected network with non-linearity (GELU).         |
        # First layer expands the dimension by mlp_ratio (typically 4x),         |
        # and the second projects it back to d_model.                            |
        # Helps the model capture complex patterns beyond attention.             |
        # -----------------------------------------------------------------------|        
        #---------------------- GELU Activation in MLP ----------------------|
        # GELU (Gaussian Error Linear Unit) is preferred over ReLU because:  |
        # 1. It provides smoother activation and better gradient flow.       |
        # 2. It weights inputs based on probability: x * P(X ≤ x), X~N(0,1). |
        # 3. Helps in capturing subtle patterns in data.                     |
        # 4. Proven to work better in Transformer-based models (e.g., BERT). |
        # 5. It allows small negative inputs to pass through with small weights,|
        #    rather than cutting them off completely like ReLU.                 |
        # 6. Research has shown that GELU leads to slightly better performance and|
        #    convergence in deep models.
        #--------------------------------------------------------------------|


        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * self.mlp_ratio),
            nn.GELU(),
            nn.Linear(d_model * self.mlp_ratio, self.d_model)
        )
    
        #----------------------------- Mask in CLIP Encoder -----------------------------|
        # Although CLIP uses only the encoder part of the Transformer architecture,     |
        # it still requires an attention mask.                                           |
        #                                                                                |
        # Purpose of the mask:                                                           |
        # - In batches, sequences are padded to the same max_seq_length for uniformity. |
        # - The padding tokens are not real input and should not affect attention scores.|
        # - The attention mask ensures the model ignores these padded tokens during     |
        #   attention computation by assigning them very low scores (e.g., -inf).        |
        #                                                                                |
        # This is critical to prevent the model from learning from irrelevant padding.   |
        #--------------------------------------------------------------------------------|
        
    def forward(self, x, mask=None):
        # Input:
        #   x     : Tensor of shape [B, seq_len, d_model] - input sequence embeddings
        #   mask  : Optional mask tensor of shape [B, 1, 1, seq_len] or [B, seq_len] depending on attention type

        # Step 1: Apply LayerNorm before attention (Pre-LN Transformer)
        #   Output shape remains [B, seq_len, d_model]
        x_n = self.ln1(x)

        # Step 2: Multi-head self-attention with optional mask
        #   Output: contextualized representations, shape [B, seq_len, d_model]
        x_n = self.mha(x_n, mask=mask)

        # Step 3: Apply second LayerNorm to the attention output
        #   Output shape: [B, seq_len, d_model]
        x_n = self.ln2(x_n)

        # Step 4: Apply MLP (usually FeedForward block) and add residual connection from input
        #   MLP output shape: [B, seq_len, d_model]
        #   Residual addition: original input x + transformed x_n
        x = x + self.mlp(x_n)

        # Final output:
        #   x : Tensor of shape [B, seq_len, d_model]
        return x
