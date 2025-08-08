
from attentionHead import AttentionHead

class MultiHeadAttention(nn.Module):
    # ------------------------------------------------------------
    # 🧠 Multi-Head Attention Diagram Idea:
    #
    #         Input Embedding (e.g., d_model = 512)
    #                         ↓
    #             Linear Projections → Q, K, V
    #                         ↓
    #          Split into multiple heads (e.g., 8)
    #
    #      ↙         ↓         ↓         ↓         ↘
    #   Head 1    Head 2    Head 3    ...       Head 8
    #   (Each head: qkv_dim = d_model // n_heads = 64)
    #
    #   → Each head performs scaled dot-product attention:
    #     softmax((QKᵗ) / sqrt(d_k)) · V
    #
    #      ↘         ↓         ↓         ↓         ↙
    #     Concatenate all head outputs → (B, seq_len, d_model)
    #                         ↓
    #             Final Linear Projection
    #
    # ✅ Result: Each head captures different aspects of input!
    # ------------------------------------------------------------

    def __init__(self, d_model, n_heads):
        super().__init__()
        # d_model --> embed dimension 
        # n_heads --> number of heads 
        
        self.d_model = d_model
        self.n_heads = n_heads
        
        self.qkv_dim = d_model//n_heads# Dimension of Q, K, V per head
        # ---------------------------------------------
        # Split total model embedding dimension (d_model)
        # evenly across all attention heads. For example:
        # if d_model = 512 and n_heads = 8,
        # then each head works with 512 // 8 = 64 dimensions.
        # This is often referred to as `head_dim` or `qkv_dim`.
        # ---------------------------------------------
        
        self.W_o = nn.Linear(d_model, d_model)# Output projection layer
        # ---------------------------------------------------------------
        # After computing attention from all heads, we concatenate their
        # outputs → shape becomes (B, seq_len, d_model).
        # This layer is a linear (dense) layer that projects the combined
        # multi-head output back to the original embedding dimension.
        #
        # It helps blend information across heads and allows the model
        # to learn how to combine the attended outputs.
        #
        # Think of it as the "final mixing" step in Multi-Head Attention.
        # ---------------------------------------------------------------
        
        self.multi_head = nn.ModuleList([AttentionHead(d_model, self.qkv_dim) for _ in range(self.n_heads)])
        # --------------------------------------------------------------------------------------------
        # Create multiple attention heads using a list comprehension.
        # Each AttentionHead has its own set of learnable weights (Q, K, V projections).
        #
        # nn.ModuleList ensures that each head is registered as a submodule,
        # so PyTorch tracks its parameters during training.
        #
        # For example, if n_heads = 4, this creates 4 parallel attention heads.
        # Each head captures different relationships or patterns from the sequence.
        # --------------------------------------------------------------------------------------------
    
    def forward(self,x, mask=None):
        
        out = torch.cat([head(x, mask = mask) for head in self.multi_head], dim = -1)
        # --------------------------------------------------------------------------------------------
        # x.shape --> [B, max_seq_len, d_model]
        # For each token in the sequence:
        # - Pass through all attention heads individually
        # - Each head returns contextual embeddings of shape [B, max_seq_len, qkv_dim]
        # - Then, concatenate the outputs from all heads along the last dimension (dim = -1)
        #   → Resulting shape becomes [B, max_seq_len, d_model] since:
        #     n_heads * qkv_dim = d_model
        #
        # This step allows the model to gather diverse information from different attention heads.
        # Each head may focus on different parts of the input sequence.
        # --------------------------------------------------------------------------------------------
        
        out = self.W_o(out)
        # --------------------------------------------------------------------------------------------
        # Project the concatenated outputs from all heads back to the original embedding dimension
        # - out shape before: [B, max_seq_len, d_model]
        # - self.W_o is a linear layer: nn.Linear(d_model, d_model)
        #
        # Why this step?
        # - Combines information from all heads
        # - Ensures the final output has the same dimension as the input embedding (d_model)
        # - Enables residual connection with the input if needed (as in Transformer blocks)
        # --------------------------------------------------------------------------------------------
        
        return out        
        # ------------------------------------------------------------------------------------------
        # Multi-Head Attention: Per-token and Matrix Form Explanation
        #
        # For each token `i` and each attention head `h`, we compute:
        #
        #   Yᵢʰ = Σⱼ (AttentionWeightᵢⱼʰ × Vⱼʰ)
        #
        # Where:
        # - AttentionWeightᵢⱼʰ = Softmax((Qᵢʰ · Kⱼʰ) / √dₖ), specific to head `h`
        # - Vⱼʰ is the value vector for token `j` in head `h`
        #
        # After computing Yᵢ for each head h = 1 to n:
        #   Yᵢ = Concat(Yᵢ¹, Yᵢ², ..., Yᵢⁿ) · Wₒ
        #
        # - Wₒ is the final output projection matrix (linear layer)
        #
        # Matrix Form (batched):
        #   Attention(Q, K, V) = Softmax(Q · Kᵗ / √dₖ) · V      → per head
        #   MultiHead(Q, K, V) = Concat(head₁, head₂, ..., headₙ) · Wₒ
        #
        # Where:
        # - Each head uses separate projection weights: W_qʰ, W_kʰ, W_vʰ ∈ [d_model, dₖ]
        # - Concatenated heads result in shape: [B, seq_len, n_heads × dₖ] == [B, seq_len, d_model]
        #
        # Intuition:
        # - Each head attends to different aspects (e.g., position, syntax, semantics)
        # - Combining multiple perspectives enriches contextual understanding of each token
        # ------------------------------------------------------------------------------------------


            
        