
import torch
import torch.nn as nn
import numpy as np

from .positionalEmbedding import PositionalEmbedding
from .transformerEncoder import TransformerEncoder
from .textEncoder import TextEncoder
from .textEncoderRetrieval import TextEncoderRetrieval


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
        
        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0,"Image Dimension should be divisible by Patch Dim"
        
        assert d_model % n_heads == 0 , "d_model should be divisible by num_heads"
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

        self.projection = nn.Parameter(torch.randn(d_model, emb_dim), requires_grad=True)
        # -----------------------------------------------------------------------------  |
        # Learnable linear projection matrix to map patch embeddings from size d_model   |
        # → emb_dim.                                                                     |
        # - Purpose: Aligns the Vision Transformer’s output embedding dimension with     |
        #   other components of the model (e.g., text encoder in CLIP).                  |
        # - Shape: [d_model, emb_dim] → multiplies each token vector to change its size. |
        # - nn.Parameter → Ensures projection weights are trainable.                     |
        # - Initialized with random values from normal distribution.                     |
        # -----------------------------------------------------------------------------  |

    def forward(self, x, mask = None):
        x = self.linear_proj(x)
        x = x.flatten(2).transpose(-2,-1)
        # --------------------------------------------------------------------------  |
        # self.linear_proj(x)                                                         |
        #   - Applies a convolution with kernel size & stride equal to the patch      |
        #     size — acting as a **patch extractor + linear projection** in one step. |
        #   - The kernel slices the image into non-overlapping patches.               |
        #   - out_channels = d_model projects each patch into a vector of size d_model|
        #   - Output shape: (B, d_model, H', W') where H'/W' = number of patches     |
        #     along height/width.                                                    |
        #                                                                            |
        # x.flatten(2)                                                               |
        #   - Flattens (H', W') into a single "patch" dimension.                     |
        #   - Shape: (B, d_model, num_patches).                                      |
        #                                                                            |
        # .transpose(-2, -1)                                                         |
        #   - Swaps patch dimension & embedding dimension to match Transformer       |
        #     expected format.                                                       |
        #   - Final shape: (B, num_patches, d_model).                                |
        # -------------------------------------------------------------------------- |
        # -------------------------------------------------------------------------- |
        # Transformers expect input as a sequence of tokens: (B, sequence_length,    |
        # embedding_dim).                                                            |
        # In ViT, each "token" is a patch embedding, so num_patches must come before |
        # the hidden dimension (d_model) in the tensor shape.                        |
        #                                                                            |
        # Example:                                                                   |
        #   Before transpose: (B, d_model, num_patches)                              |
        #   After transpose:  (B, num_patches, d_model)  ✅ Matches Transformer input|
        # -------------------------------------------------------------------------- |

        x = torch.cat(
            (self.cls_token.expand(x.shape[0],-1,-1),x),
            dim = 1
        )
        # -------------------------------------------------------------------------- |
        # Add the learnable [CLS] token at the start of the patch sequence.          |
        #                                                                            |
        # 1. self.cls_token: shape (1, 1, d_model) → learnable embedding that        |
        #    represents the "summary" token for the entire image.                    |
        # 2. .expand(B, 1, d_model): duplicates CLS token for each batch sample.     |
        # 3. torch.cat(..., dim=1): places CLS token before all patch tokens,        |
        #    making it the first token in the sequence.                              |
        #                                                                            |
        # Final shape: (B, max_seq_len, d_model), where max_seq_len = num_patches+1. |
        # This CLS token will later hold the global representation after the         |
        # Transformer layers for classification or projection.                       |
        # -------------------------------------------------------------------------- |

        x = self.positional_embeddings(x)
        
        for encoder_layer in self.transformer_encoder:
            x = encoder_layer(x, mask)
        # --------------------------------------------------------------------------  |
        # 1. Add positional embeddings to each token (CLS + patch tokens) so          |
        #    the Transformer is aware of the order/position of tokens.                |
        #    Output: (B, max_seq_len, d_model) with position info added.              |
        #                                                                             |
        # 2. Pass the sequence through n_layers of TransformerEncoder blocks:         |
        #    - Each layer applies multi-head self-attention + feed-forward network.   |
        #    - Uses residual connections and layer normalization.                     |
        #    - mask (if given) prevents attention to padding tokens.                  |
        #                                                                             |
        # Effect: Gradually builds richer contextual representations of the           |
        # sequence, allowing each patch to attend to every other patch.               |
        # --------------------------------------------------------------------------  |
        
        x = x[:,0,:]
        # ----------------------------------------------------------------------------|
        # Select the CLS token embedding from the Transformer output:                 |
        # - x[:, 0, :] → takes only the first token (CLS) for each batch.             |
        # - CLS token acts as a learnable "summary" of the entire image,              |
        #   gathering information from all patches via self-attention.                |
        # - Output shape: (B, d_model).                                               |
        # --------------------------------------------------------------------------  |

        if self.projection is not None:
            x = x @ self.projection
        # ------------------------------------------------------------------------- |
        # Project CLS token embedding to target embedding dimension                 |
        # After processing through the Transformer, the CLS token is a vector of    |
        # size d_model.                                                             |
        # If emb_dim != d_model, we need to map it into the new space for           |
        # downstream tasks (e.g., matching image embeddings with text embeddings    |
        # in CLIP).                                                                 |
        # The learnable projection matrix (shape: [d_model, emb_dim]) acts like an  |
        # "exchange rate table", converting features from the d_model space into    |
        # the emb_dim space.                                                        |
        # This is done via a matrix multiplication:                                 |
        #   (B, d_model) @ (d_model, emb_dim) → (B, emb_dim)                        |
        # The weights are learned so that the projected embeddings align well with  |
        # the target space.                                                         |
        # ------------------------------------------------------------------------- |
        
        x = x / torch.norm(x, dim = -1, keepdim = True)
        # ------- L2 Normalization of Embeddings -------|
        # Computes the Euclidean (L2) norm for each embedding vector:
        #     norm = sqrt(sum(x_i^2))
        # Divides each vector by its L2 norm so that its magnitude becomes exactly 1.
        # This places all embeddings on the surface of a unit hypersphere.
        #
        # Why? → Ensures similarity comparisons depend only on direction, not magnitude.
        # Common in cosine similarity, contrastive learning, and retrieval tasks.
        #
        # Visual Analogy:
        # Before L2 Norm:   -->  Different lengths, same or different directions
        #    ↑      ↗
        #    |    ↗
        #
        # After L2 Norm:   -->  All vectors have length = 1, on the unit sphere
        #    ↗      ↗
        #   /      /
        #  •------•------•
        #         center
        # Now only the angle (direction) matters for similarity.                  |
        
        return x







  



        