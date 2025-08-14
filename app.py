# Vision
emb_dim = 128 
vit_d_model = 32 # vit_heads * vit_layers = vit_d_model
img_size = (80,80)
patch_size = (5,5) 
n_channels = 3
vit_layers = 8
vit_heads = 4 

# Text
vocab_size = 256
text_d_model = 64 #  -->  text_heads * text_layers = text_d_model
max_seq_length = 128
text_heads = 8
text_layers = 8
lr = 1e-3
epochs = 50
batch_size = 128



import gradio as gr
import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader

from src.clip import CLIP
from src.Tokenization import tokenizer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps
import pdb
import os
from tqdm import tqdm
import random

import warnings

from src.dataset import MyntraDataset
from src.Tokenization import tokenizer
from src.clip import CLIP

# ------------------------------------------------------------------------
# Vision
emb_dim = 128 
vit_d_model = 32 # vit_heads * vit_layers = vit_d_model
img_size = (80,80)
patch_size = (5,5) 
n_channels = 3
vit_layers = 8
vit_heads = 4 

# Text
vocab_size = 256
text_d_model = 64 #  -->  text_heads * text_layers = text_d_model
max_seq_length = 128
text_heads = 8
text_layers = 8
lr = 1e-3
epochs = 100
batch_size = 128

# ------------------------------------------------------------------------

df = pd.read_csv('Dataset/myntradataset/styles.csv', usecols = ['id', 'subCategory'])

unique, counts = np.unique(df["subCategory"].tolist(), return_counts = True)
# print(f"Classes: {unique}: {counts}")

# Split the dataset into training and validation sets
train_df, val_df = train_test_split(df, test_size=0.10, random_state=42)

# Print the sizes of the datasets
print(f"Train size: {len(train_df)}, Validation size: {len(val_df)}")
class_names = df['subCategory'].unique()
class_names = [str(name).lower() for name in class_names]

# Replace in-place
for i, name in enumerate(class_names):
    if name == "lips":
        class_names[i] = "lipstick"
    elif name == "eyes":
        class_names[i] = "eyelash"
    elif name == "nails":
        class_names[i] = "nail polish"

captions = {idx: class_name for idx, class_name in enumerate(class_names)}

# for idx, caption in captions.items():
#     print(f"{idx}: {caption}\n")  
    
    

#Create datasets
train_dataset = MyntraDataset(data_frame=train_df ,captions = captions, target_size =80)
val_dataset = MyntraDataset(data_frame=val_df ,captions = captions ,target_size =80)
test_dataset = MyntraDataset(data_frame=val_df, captions = captions, target_size = 224)


print("Number of Samples in Train Dataset:", len(train_dataset))
print("Number of Samples in Validation Dataset:", len(val_dataset))


train_loader = DataLoader(train_dataset, shuffle = True, batch_size = batch_size,num_workers = 5)
val_loader  = DataLoader(val_dataset, shuffle = False, batch_size = batch_size,num_workers = 5)
test_loader = DataLoader(test_dataset, shuffle = False, batch_size = batch_size, num_workers = 5)

# Load the model and tokenizer
retrieval_model = CLIP(emb_dim, vit_layers, vit_d_model, img_size, patch_size, n_channels, vit_heads, vocab_size, max_seq_length, text_heads, text_layers, text_d_model, retrieval=True).to(device)
retrieval_model.load_state_dict(torch.load("clip.pt", map_location=device))

# Function to process the query and return the top 30 images
def retrieve_images(query):
    query_text, query_mask = tokenizer(query)
    query_text = query_text.unsqueeze(0).to(device)  # Add batch dimension
    query_mask = query_mask.unsqueeze(0).to(device)

    with torch.no_grad():
        query_features = retrieval_model.text_encoder(query_text, mask=query_mask)
        query_features /= query_features.norm(dim=-1, keepdim=True)

    # Step 2: Encode all images in the dataset and store features
    image_features_list = []
    image_paths = []

    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=5)

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            features = retrieval_model.vision_encoder(images)
            features /= features.norm(dim=-1, keepdim=True)
            
            image_features_list.append(features)
            image_paths.extend(batch["id"])  # Assuming batch contains image paths or IDs

    # Concatenate all image features
    image_features = torch.cat(image_features_list, dim=0)

    # Step 3: Compute similarity using the CLIP model's logic
    similarities = (query_features @ image_features.T) * torch.exp(retrieval_model.temperature)
    similarities = similarities.softmax(dim=-1)

    # Retrieve top 30 matches
    top_values, top_indices = similarities.topk(30)

    # Step 4: Retrieve and display top N images
    images_to_display = []
    for value, index in zip(top_values[0], top_indices[0]):
        img_path = image_paths[index]
        img = Image.open(img_path).convert("RGB")
        images_to_display.append(np.array(img))

    return images_to_display

# Define the Gradio interface
def gradio_app(query):
    images = retrieve_images(query)
    return images

import gradio as gr

# Create Gradio Interface
with gr.Blocks() as interface:
    # Centered title
    gr.Markdown("<h1 style='text-align: center;'> 👒 Image Retrieval with CLIP -  👔👖 E-commerce Fashion 👚🥻</h1>")
    
    with gr.Row():
        # Textbox for query input
        query_input = gr.Textbox(placeholder="Enter your search query...", show_label=False, elem_id="custom-textbox")
        
        # Small submit button
        submit_btn = gr.Button("Search 🔍", elem_id="small-submit-btn")
    
    # Gallery output for displaying images
    gallery_output = gr.Gallery(label="Top 30 Matches").style(grid=[8], container=True)
    
    # Link the submit button to the function
    submit_btn.click(fn=gradio_app, inputs=query_input, outputs=gallery_output)

    # Custom CSS to make the submit button small and increase the font size in the textbox
    gr.HTML("""
    <style>
    #small-submit-btn {
        padding: 0.5rem 1rem;
        font-size: 0.8rem;
    }
    #custom-textbox input {
        font-size: 1.5rem;
    }
    </style>
    """)

# Launch the app
interface.launch()
