import torch
import torch.nn as nn
import numpy as np

from .VITEncoder import VisionEncoder
from .textEncoder import TextEncoder
from .textEncoderRetrieval import TextEncoderRetrieval

class CLIP(nn.Module):
    """
    --- 
    CLIP Model Implementation
    ---
    A PyTorch implementation of the CLIP (Contrastive Language–Image Pre-training) model, 
    which encodes both images and text into a shared embedding space. 
    This class builds separate image and text encoders (ViT for images, Transformer for text) 
    and allows for retrieval tasks by computing similarity between image and text embeddings.

    ---
    Parameters
    ---
    emb_dim : int
        The final embedding dimension for both image and text features.
    vit_layers : int
        Number of Transformer encoder layers in the Vision Transformer.
    vit_d_model : int
        Dimension of each hidden layer in the Vision Transformer.
    img_size : int
        Input image size (assumed square).
    patch_size : int
        Size of each patch for patch embedding in the Vision Transformer.
    n_channels : int
        Number of channels in the input image (e.g., 3 for RGB).
    vit_heads : int
        Number of attention heads in the Vision Transformer.
    vocab_size : int
        Number of unique tokens in the text vocabulary.
    max_seq_length : int
        Maximum number of tokens in a text sequence.
    text_heads : int
        Number of attention heads in the Text Transformer.
    text_layers : int
        Number of Transformer encoder layers in the Text Transformer.
    text_d_model : int
        Dimension of each hidden layer in the Text Transformer.
    retrieval : bool, optional
        If True, the model is optimized for retrieval tasks (image-to-text or text-to-image). 
        Default is False.

    ---
    Returns
    ---
    nn.Module
        A CLIP model instance ready for training or inference.

    ---
    Example
    ---
    # >>> model = CLIP(
    # ...     emb_dim=512, vit_layers=12, vit_d_model=768,
    # ...     img_size=224, patch_size=16, n_channels=3, vit_heads=12,
    # ...     vocab_size=49408, max_seq_length=77, text_heads=8, 
    # ...     text_layers=12, text_d_model=512, retrieval=True
    # ... )
    # >>> img_emb, text_emb = model(image_batch, text_batch)
    # >>> similarity = torch.matmul(img_emb, text_emb.T)
    """
    def __init__(self, emb_dim, vit_layers, vit_d_model, img_size, patch_size, n_channels,
                 vit_heads, vocab_size, max_seq_length, text_heads, text_layers, text_d_model,
                 retrieval = False
                 ):
        super().__init__()
        # ╔════════════════════════════════════════════════════════════════════════════════════╗
        # ║ Create the Vision Encoder module                                                   ║
        # ╠════════════════════════════════════════════════════════════════════════════════════║
        # ║ • self.vision_encoder                                                              ║
        # ║     - This is an instance of the VisionEncoder class, which processes images.      ║
        # ║                                                                                    ║
        # ║ • Parameters passed:                                                               ║
        # ║     1. vit_d_model   → Dimensionality of the Vision Transformer (ViT) hidden layer ║
        # ║     2. img_size      → Size of the input image (e.g., 224x224)                     ║
        # ║     3. patch_size    → Size of each patch the image will be split into             ║
        # ║     4. n_channels    → Number of channels in the image (e.g., 3 for RGB)           ║
        # ║     5. vit_heads     → Number of attention heads in ViT                            ║
        # ║     6. vit_layers    → Number of transformer layers in ViT                         ║
        # ║     7. emb_dim       → Final embedding dimension after encoding                    ║
        # ║                                                                                    ║
        # ║ • Purpose:                                                                         ║
        # ║     The VisionEncoder takes an image, splits it into patches, applies a Vision     ║
        # ║     Transformer, and outputs a fixed-length embedding vector representing the      ║
        # ║     visual features of the image.                                                  ║
        # ╚════════════════════════════════════════════════════════════════════════════════════╝

        self.vision_encoder = VisionEncoder(vit_d_model, img_size, patch_size, n_channels, vit_heads,
                                            vit_layers, emb_dim)
        # ╔══════════════════════════════════════════════════════════════════════════════════════╗
        # ║ Purpose:                                                                             ║
        # ║     Initializes the text encoder component of the CLIP model.                        ║
        # ║                                                                                      ║
        # ║ Details:                                                                             ║
        # ║     • Uses different text encoder architectures depending on whether retrieval mode  ║
        # ║       is enabled.                                                                    ║
        # ║     • If `retrieval` is True → Uses `TextEncoderRetrieval` for retrieval-optimized   ║
        # ║       embeddings (better suited for search & matching tasks).                        ║
        # ║     • If `retrieval` is False → Uses `TextEncoder` for standard CLIP text encoding.  ║
        # ║                                                                                      ║
        # ║ Parameters passed to the encoder:                                                    ║
        # ║     vocab_size      → Size of the text vocabulary.                                   ║
        # ║     text_d_model    → Dimensionality of the text model.                              ║
        # ║     max_seq_length  → Maximum number of tokens in input text.                        ║
        # ║     text_layers     → Number of transformer layers for text encoding.                ║
        # ║     text_heads      → Number of attention heads in text transformer.                 ║
        # ║     emb_dim         → Output embedding dimension (shared with vision encoder).       ║
        # ║                                                                                      ║
        # ║ Effect:                                                                              ║
        # ║     Creates `self.text_encoder` as either retrieval-specialized or standard encoder, ║
        # ║     ensuring its output is compatible with the vision encoder for multimodal         ║
        # ║     alignment.                                                                       ║
        # ╚══════════════════════════════════════════════════════════════════════════════════════╝

        if retrieval:
            self.text_encoder = TextEncoderRetrieval(vocab_size, text_d_model, max_seq_length,
                                                   text_layers, text_heads, emb_dim)
        else:
            self.text_encoder = TextEncoder(vocab_size, text_d_model, max_seq_length,
                                            text_layers, text_heads, emb_dim)
        # ╔═══════════════════════════════════════════════════════════════════════════╗
        # ║ Define Learnable Temperature Parameter for Contrastive Loss               ║
        # ╠═══════════════════════════════════════════════════════════════════════════╣
        # ║ - Creates a scalar tensor initialized with log(1 / 0.07) ≈ 2.659.         ║
        # ║ - This scalar is wrapped as an `nn.Parameter`, making it trainable.       ║
        # ║ - Used in contrastive learning to scale logits before softmax.            ║
        # ║ - Lower temperature → sharper probability distribution.                   ║
        # ║ - Higher temperature → smoother probability distribution.                 ║
        # ║ - Allowing it to be learned helps the model adaptively tune similarity    ║
        # ║   scoring between embeddings.                                             ║
        # ╚═══════════════════════════════════════════════════════════════════════════╝
        self.temperature = nn.Parameter(torch.ones([])*np.log(1/0.07))
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def CLIPLoss(self, logits, device = None):
        # ╔══════════════════════════════════════════════════════════════════════════════════════╗
        # ║ CLIP Symmetric Contrastive Loss (InfoNCE-style)                                      ║
        # ╠══════════════════════════════════════════════════════════════════════════════════════╣
        # ║ Inputs/Shapes                                                                        ║
        # ║ • logits:  [B, B] similarity matrix; logits[i, j] = sim(image_i, text_j).            ║
        # ║ • device:  where to place target labels (e.g., "cuda").                              ║
        # ║ • B:       batch size (number of aligned image–text pairs).                          ║
        # ║                                                                                      ║
        # ║ Targets                                                                              ║
        # ║ • labels = [0, 1, 2, ..., B-1]. For row i, the correct column is i (diagonal).       ║
        # ║   This encodes that image_i matches text_i and all others are negatives.             ║
        # ║                                                                                      ║
        # ║ Loss Terms                                                                           ║
        # ║ • loss_t = CE(logits, labels):    image → text (row-wise classification).            ║
        # ║ • loss_v = CE(logits.T, labels):  text → image (column-wise classification).         ║
        # ║ • loss   = (loss_t + loss_v) / 2: symmetric average to train both directions.        ║
        # ║                                                                                      ║
        # ║ Intuition                                                                            ║
        # ║ • For each image (row), push the matching text (diagonal) to have the highest score. ║
        # ║ • For each text (row in logits.T), push the matching image to the top as well.       ║
        # ║ • Off-diagonal entries are negatives and are pushed down by softmax.                 ║
        # ║                                                                                      ║
        # ║ Practical Notes                                                                      ║
        # ║ • logits should be *unnormalized* similarities (often scaled by a learnable temp).   ║
        # ║ • CrossEntropyLoss applies log-softmax internally; default reduction is 'mean'.      ║
        # ║ • Ensure batches are aligned: (image_i, text_i) is the only positive per row/col.    ║
        # ║ • For multi-positive settings, standard CE isn’t sufficient (use soft targets/etc.). ║
        # ╚══════════════════════════════════════════════════════════════════════════════════════╝
        
        labels = torch.arange(logits.shape[0]).to(device)
        
        loss_v = nn.functional.cross_entropy(logits.transpose(-2,-1), labels)
        loss_t = nn.functional.cross_entropy(logits, labels)
        
        loss = (loss_v + loss_t) / 2

        return loss
    
    def forward(self, image, text, mask  = None):
        """
        ╔═════════════════════════════════════════════════════════════════════════════╗
        ║ forward(self, image, text, mask=None)                                       ║
        ║ ----------------------------------------------------------------------------║
        ║ PURPOSE:                                                                    ║
        ║ Runs a forward pass of the CLIP model:                                      ║
        ║   1. Encode image and text into feature embeddings.                         ║
        ║   2. Compute similarity scores between all image-text pairs.                ║
        ║   3. Scale similarities with a learnable temperature parameter.             ║
        ║   4. Compute the symmetric CLIP loss (image→text and text→image).           ║
        ║                                                                             ║
        ║ PARAMETERS:                                                                 ║
        ║   image : Tensor                                                            ║
        ║       Input batch of images. Shape: (batch_size, C, H, W)                   ║
        ║   text : Tensor                                                             ║
        ║       Input batch of tokenized text sequences. Shape: (batch_size, seq_len) ║
        ║   mask : Tensor or None (optional)                                          ║
        ║       Optional attention mask for text encoder (not used in this snippet).  ║
        ║                                                                             ║
        ║ RETURNS:                                                                    ║
        ║   loss : Tensor (scalar)                                                    ║
        ║       The average symmetric contrastive loss for the batch.                 ║
        ║                                                                             ║
        ║ PROCESS:                                                                    ║
        ║   Step 1 → Encode images using vision_encoder → V_e                         ║
        ║   Step 2 → Encode text using text_encoder → T_e                             ║
        ║   Step 3 → Compute similarity matrix: V_e @ T_eᵀ                            ║
        ║   Step 4 → Scale similarities by exp(temperature)                           ║
        ║   Step 5 → Compute CLIP loss (image→text and text→image)                    ║
        ╚═════════════════════════════════════════════════════════════════════════════╝
    """

        V_e = self.vision_encoder(image)
        T_e = self.text_encoder(text, mask)
        
        logits = (V_e @ T_e.transpose(-2,-1)) * torch.exp(self.temperature)
        
        loss = self.CLIPLoss(logits, self.device)
        
        return loss
            
            
            