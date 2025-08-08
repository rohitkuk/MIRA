
class TransformerEncoder(self):
    
    def __init__(self, d_model, n_heads, mlp_ratio = 4)
        
        """
        
        Args:
        d_model (int):      Dimensionality of input & output embeddings (e.g. 512 or 768).
        n_heads (int):      Number of parallel attention heads (e.g. 8 or 12).
        mlp_ratio (float):  Multiplier for hidden size of the position-wise feed-forward layer.
                                If mlp_ratio=4 and d_model=512, then hidden dim = 512 * 4 = 2048.
        """
    super().__init__()
    self.d_model = d_model
    self.n_heads = n_heads
    self.mlp_ratio = mlp_ratio
    
    