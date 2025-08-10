class VisionEncoder(nn.Module):
    # ---------------------------------------------------------
    # VisionEncoder (ViT-style):
    # Args:
    #   d_model     → Embedding dimension for transformer layers
    #   img_size    → Size of input image (assumes square: img_size x img_size)
    #   patch_size  → Size of image patches (assumes square: patch_size x patch_size)
    #   n_channels  → Number of image channels (e.g., 3 for RGB)
    #   n_heads     → Number of attention heads in each Transformer encoder
    #   n_layers    → Number of stacked Transformer encoder layers
    #   emb_dim     → Final output embedding dimension for the image representation
    #
    # Purpose:
    #   - Splits image into patches
    #   - Embeds patches into vectors of size d_model
    #   - Passes sequence of patch embeddings through Transformer encoders
    #   - Produces a global representation (e.g., [CLS] token) for the image
    # ---------------------------------------------------------

    def __init__(self, d_model, img_size, patch_size, n_channels,
                 n_heads, n_layers, emb_dim):
        super().__init__()
        
        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0,
        "Image Dimension should be divisible by Patch Dim"
        
        assert d_model % n_heads = 0 , "d_model should be divisible by num_heads"
        # ---------------------------------------------------------
        # Validations:
        # 1. Image size must be divisible by patch size in both height and width
        #    → Ensures the image can be evenly split into patches without leftovers
        # 2. d_model must be divisible by n_heads
        #    → Ensures each attention head gets an equal share of the embedding dimensions
        # ---------------------------------------------------------
        
        self.num_patches = (img_size[0] * img_size[1])//(patch_size[0]*patch_size[1])
        # -------------------------------------------------------------------|
        # Calculate total number of patches (max_seq_length)                 |
        # -------------------------------------------------------------------|
        # Splits the input image into non-overlapping patches.               |
        # Formula: num_patches = (image_height × image_width) ÷              |
        #                        (patch_height × patch_width)                |
        # This gives the number of "tokens" the image becomes when           |
        # represented in the Transformer.                                    |
        # Example:                                                           |
        #   img_size = (224, 224), patch_size = (16, 16)                     |
        #   num_patches = (224×224) ÷ (16×16) = 50176 ÷ 256 = 196 patches    |
        # Each patch will be flattened and projected to a d_model-dimensional|
        # embedding. This count becomes the max_seq_length for               |
        # positional embeddings.                                             |
        # -------------------------------------------------------------------|

        self.max_seq_length = self.num_patches+1
        # -------------------------------------------------------------------|
        # Define max_seq_length for the Transformer                          |
        # -------------------------------------------------------------------|
        # Adds 1 to num_patches to account for the special [CLS] token.      |
        # [CLS] token acts as a learnable embedding prepended to the         |
        # sequence of patch embeddings, used to aggregate the global         |
        # representation of the image.                                       |
        # Formula: max_seq_length = num_patches + 1                          |
        # Example:                                                           |
        #   num_patches = 196 → max_seq_length = 196 + 1 = 197               |
        # This total length is used for positional embeddings and            |
        # sequence processing in the encoder.                                |
        # -------------------------------------------------------------------|
        
        self.linear_proj = nn.Conv2d(
            in_channels = n_channels, out_channels = d_model,
            kernel_size = patch_size[0], stride = patch_size[0]
            )
        # -----------------------------------------------------------------------------  |
        # Patch Embedding Projection using Conv2d                                        |
        # -----------------------------------------------------------------------------  |
        # Purpose:                                                                       |
        #   - Split the input image into non-overlapping patches.                        |
        #   - Flatten each patch and project it into a d_model-dimensional embedding.    |
        #   - This is done in a single operation using Conv2d instead of manual slicing. |
        #                                                                                |
        # Parameters:                                                                    |
        #   - in_channels  = n_channels (e.g., 3 for RGB images)                         |
        #   - out_channels = d_model (size of embedding vector for each patch)           |
        #   - kernel_size  = patch_size[0] (height & width of each patch)                |
        #   - stride       = patch_size[0] (ensures patches do not overlap)              |
        #                                                                                |
        # How it works:                                                                  |
        #   - The Conv2d kernel slides over the image with step size equal to patch size |
        #   - Each kernel "sees" exactly one patch at a time                             |
        #   - Instead of outputting a single scalar, it outputs a vector of length d_model
        #                                                                                |
        # Example:                                                                       |
        #   Input image: [B, 3, 224, 224] (RGB)                                          |
        #   patch_size: (16, 16)                                                         |
        #   After Conv2d: [B, d_model, 14, 14]                                           |
        #   (14 × 14 = 196 patches total)                                                |
        #                                                                                |
        # Why Conv2d is used here:                                                       |
        #   - Much faster than manually reshaping + Linear projection                    |
        #   - Achieves the same effect as "flatten patch + Linear layer"                 |
        #                                                                                |
        # Output shape:                                                                  |
        #   [B, d_model, num_patches_h, num_patches_w]                                   |
        #   where:                                                                       |
        #       num_patches_h = img_height / patch_height                                |
        #       num_patches_w = img_width  / patch_width                                 |
        # -----------------------------------------------------------------------------  |

        self.cls_token = nn.Parameter(
            torch.randn(1,1,d_model),
            requires_grad = True
            )
        # ----------------------------------------------------------------------------- |
        # Learnable Class Token ([CLS])                                                 |
        # ----------------------------------------------------------------------------- |
        # Purpose:                                                                      |
        #   - A special token added to the beginning of the patch embedding sequence.   |
        #   - Acts as a global representation of the entire image.                      |
        #   - After Transformer processing, this token’s final embedding is used for    |
        #     classification or other global tasks.                                     |
        #                                                                               |
        # Shape:                                                                        |
        #   - Initialized as: [1, 1, d_model]                                           |
        #       1 → single token                                                        |
        #       1 → batch dimension placeholder (will be repeated per batch)            |
        #       d_model → embedding dimension                                           |
        #                                                                               |
        # How it works in forward pass:                                                 |
        #   - The [CLS] token is concatenated in front of all patch embeddings          |
        #   - Positional embeddings are also applied to it                              |
        #   - Self-attention allows it to attend to all patch tokens and vice versa     |
        #                                                                               |
        # Example:                                                                      |
        #   Patch embeddings shape before:  [B, num_patches, d_model]                   |
        #   After adding CLS: [B, num_patches + 1, d_model]                             |
        #                                                                               |
        # Initialization:                                                               |
        #   - torch.randn(...) → Random normal values                                   |
        #   - requires_grad=True → Updated during training                              |
        # ----------------------------------------------------------------------------- |
        
        # ----------------------------------------------------------------------------- |
        # Why use nn.Parameter for CLS token?                                           |
        # ----------------------------------------------------------------------------- |
        # 1. nn.Parameter registers the tensor as a learnable parameter of the model.   |
        # 2. This means:                                                                |
        #      - It appears in model.parameters()                                       |
        #      - It gets updated automatically during training (via backpropagation).   |
        # 3. A normal torch.Tensor without nn.Parameter would be treated as constant,   |
        #    unless manually added to optimizer's parameter list.                       |
        # 4. For CLS token:                                                             |
        #      - Starts as random vector                                                |
        #      - Learns to store global image information through training              |
        # ----------------------------------------------------------------------------- |

        self.positional_embeddings= PositionalEmbedding(d_model, self.max_seq_length)
        self.transformer_encoder = nn.ModuleList([
            TransformerEncoder(d_model, n_heads) for _ in range(n_layers)
            
        ])
        # ----------------------------------------------------------------------------- |
        # n_layers → Number of Transformer Encoder blocks stacked sequentially.         |
        # - Each layer applies self-attention + feedforward transformation, allowing    |
        #   the model to capture increasingly abstract and global relationships.        |
        # - More layers = deeper reasoning ability, but higher computational cost.      |
        # - Acts like "depth" in a CNN: shallow layers capture low-level patterns,      |
        #   deeper layers capture high-level, semantic relationships.                   |
        # ----------------------------------------------------------------------------- |

        self.projection = nn.Parameter(torch.randn(d_model, emb_dim), requires_grad_True)
        # -----------------------------------------------------------------------------  |
        # Learnable linear projection matrix to map patch embeddings from size d_model   |
        # → emb_dim.                                                                     |
        # - Purpose: Aligns the Vision Transformer’s output embedding dimension with     |
        #   other components of the model (e.g., text encoder in CLIP).                  |
        # - Shape: [d_model, emb_dim] → multiplies each token vector to change its size. |
        # - nn.Parameter → Ensures projection weights are trainable.                     |
        # - Initialized with random values from normal distribution.                     |
        # -----------------------------------------------------------------------------  |



        