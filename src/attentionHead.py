# For now I will put everythign in the main file and once done will move them to the individual files. 

# All the comments below are generated from ChatGPT.




# Imports
import torch
import torch.nn as nn

import torch.optim as optim
import torchvision 
import torchvision.transforms as T

from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps
 
import pdb
import os
from tqdm import tqdm
 
import warnings
warnings.filterwarnings('ignore')


def tokenizer(text):
    pass


class AttentionHead(nn.Module):
    def __init__(self, d_model, qkv_dim):
        super().__init__() #calling the constructor of the base class
        self.qkv_dim = qkv_dim
        self.query = nn.Linear(d_model, self.qkv_dim) 
        self.key   = nn.Linear(d_model, self.qkv_dim)
        self.value = nn.Linear(d_model, self.qkv_dim)
        
    def forward(self, x, mask = None):
        # x.shape --> [B, max_seq_len, d_model]  
        # B: Batch size (e.g., 32), max_seq_len: Number of tokens or patches (e.g., 128 for text, 196 for ViT), 
        # d_model: Embedding size per token (e.g., 64, 512, 768)
        Q = self.query(x) #[B,max_seq_len,vit_heads/qkv_dim] not sure, smae for below
        K = self.key(x)
        V = self.value(x)
        
        # ------------------------------------------------------------
        # Matrix Multiplication Rules for Attention:  
        #                                             
        # X: (a, b) @ Y: (b, c) --> Result: (a, c)     
        #                                             
        # Batched Matrix Multiplication:              
        # X: (B, a, b) @ Y: (B, b, c) --> (B, a, c)    
        #                                             
        # - B: Batch size (remains unchanged)         
        # - Inner dimensions must match (b == b)      
        # - Output shape: outer dims → (a, c)         
        #                                             
        # Example in Attention:                       
        # Q: (B, seq_len, d_k)                        
        # K.T: (B, d_k, seq_len)                      
        # Q @ K.T --> (B, seq_len, seq_len)           
        # ------------------------------------------------------------



        attention = Q @ K.transpose(-2,-1)
        # ------------------------------------------------------------

        # Example with tokens to understand Q @ K.T in Attention  
        #                                                         
        # Suppose we have 3 tokens: ["I", "love", "pizza"]        
        # Each token gets a Query (Q) and Key (K) vector of size 2
        #                                                         
        # Q = [                                                   
        #   [1, 0],   # Q for "I"                                 
        #   [0, 1],   # Q for "love"                              
        #   [1, 1],   # Q for "pizza"                             
        # ]                                                       
        #                                                         
        # K = [                                                   
        #   [1, 0],   # K for "I"                                 
        #   [0, 1],   # K for "love"                              
        #   [1, 1],   # K for "pizza"                             
        # ]                                                       
        #                                                         
        # Q @ K.T results in a (3, 3) matrix of dot products:     
        #                                                         
        #         "I"   "love"  "pizza"     ← Keys                
        #       ------------------------                          
        # "I"   |  1      0       1        |                      
        # "love"|  0      1       1        | ← Queries            
        # "pizza"| 1      1       2        |                      
        #                                                         
        # - Each row = attention scores for a query token         
        # - Each col = how much that token is attended to         
        # - These scores are later scaled and passed to softmax   
        # ------------------------------------------------------------


        
        
        # scaled dot-product attention.
        attention = attention/self.qkv_dim**0.5 #Dividing by Square root of QKV_DIM to scale
        # ------------------------------------------------------------
        # Scaling Attention Scores                                
        #                                                         
        # We scale the dot product Q·Kᵗ by sqrt(d_k)              
        # Formula: attention = (Q @ K.T) / sqrt(d_k)              
        #                                                         
        # Why?                                                    
        # - Dot products can become large if d_k (vector size) is 
        #   big, leading to very sharp softmax values             
        # - This causes gradients to vanish or explode            
        #                                                         
        # Scaling by sqrt(d_k) keeps values in a balanced range   
        # → leads to better gradient flow and training stability  
        #                                                         
        # Example: if d_k = 64 → scale by sqrt(64) = 8            
        # ------------------------------------------------------------



        if mask is not None:
            mask = attention.masked_fill(mask==0, float('-inf'))
        # ------------------------------------------------------------

        # Apply Attention Mask for Padded Tokens                  
        #                                                         
        # - `mask` contains 1s for valid tokens and 0s for padding
        # - We replace attention scores at padding positions with 
        #   `-inf` using `masked_fill()`                          
        #                                                         
        # - Why?                                                  
        #   Softmax(Q·Kᵗ) turns `-inf` to 0 → ignores padding     
        #   Ensures attention focuses only on real tokens         
        #                                                         
        # Example:                                                
        #   mask: [1, 1, 0] → 3rd token is padding                
        #   attention: [0.5, 0.3, -inf] → softmax ignores 3rd     
        # ------------------------------------------------------------


        
        attention = torch.softmax(attention, dim = -1) #along last dim  # (softmax(Q_K^T)/sqrt(d_k)).V -->  [B,max_seq_len,max_seq_len]
        attention = attention @ V # [B,max_seq_len,max_seq_len]
        # ------------------------------------------------------------
        # Compute Final Attention Output                                 
        #                                                                
        # 1. Softmax over attention scores                               
        #    - Shape: (B, seq_len, seq_len)                              
        #    - Converts dot-product scores into attention weights        
        #    - For each query token: how much to attend to each key     
        #    - `dim = -1` → softmax along keys for each query           
        #                                                                
        # 2. Multiply attention weights with Value vectors               
        #    - V: (B, seq_len, d_v)                                      
        #    - Output: (B, seq_len, d_v)                                 
        #    - Each token's output is a weighted sum of value vectors   
        #      from all tokens                                           
        #                                                                
        #    Final result: Contextual representation of tokens          
        # ------------------------------------------------------------

        return attention #Y_i
        # -----------------------------------------------------------------------------------------------
        # Final output Yᵢ: Contextual embedding for each token after self-attention
        #
        # Each token's new representation Yᵢ captures information from all other tokens in the sequence
        # based on learned attention scores — this is called a *contextual embedding*.
        #
        # Formula for each token i:
        #   Yᵢ = Σⱼ (AttentionWeightᵢⱼ × Vⱼ)
        #
        # Where:
        # - AttentionWeightᵢⱼ = Softmax((Qᵢ · Kⱼ) / √dₖ)
        # - Vⱼ is the value vector of token j
        #
        # Intuition:
        # - Yᵢ is a weighted sum of all other tokens' value vectors (including itself)
        # - The weights determine how relevant each token j is to token i
        #
        # Matrix form for the full batch:
        #   Y = Softmax(Q · Kᵗ / √dₖ) · V
        #
        # Output shape: (B, seq_len, d_v)
        # - B: Batch size
        # - seq_len: Number of tokens
        # - d_v: Dimension of each value vector (usually = qkv_dim)
        #
        # ✅ This is the heart of the Transformer: each token gets a new vector (Yᵢ) that encodes
        #    not just its own meaning, but also its *context* — how it relates to every other token.
        # ---------------------------------------------------------------------------------------------------
