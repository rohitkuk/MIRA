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
epochs = 1
batch_size = 128

# ------------------------------------------------------------------------

df = pd.read_csv('Dataset/myntradataset/styles.csv', usecols = ['id', 'subCategory'])

unique, counts = np.unique(df["subCategory"].tolist(), return_counts = True)
print(f"Classes: {unique}: {counts}")

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

for idx, caption in captions.items():
    print(f"{idx}: {caption}\n")  
    
    

#Create datasets
train_dataset = MyntraDataset(data_frame=train_df ,captions = captions, target_size =80)
val_dataset = MyntraDataset(data_frame=val_df ,captions = captions ,target_size =80)
test_dataset = MyntraDataset(data_frame=val_df, captions = captions, target_size = 224)


print("Number of Samples in Train Dataset:", len(train_dataset))
print("Number of Samples in Validation Dataset:", len(val_dataset))


train_loader = DataLoader(train_dataset, shuffle = True, batch_size = batch_size,num_workers = 5)
val_loader  = DataLoader(val_dataset, shuffle = False, batch_size = batch_size,num_workers = 5)
test_loader = DataLoader(test_dataset, shuffle = False, batch_size = batch_size, num_workers = 5)

# Function to visualize a batch of images with their class names
def visualize_samples(dataset, tokenizer, num_samples=5):
    # Get a batch of samples
    for i in range(num_samples):
        sample = dataset[i]  # Access the ith sample in the dataset
        image = sample['image']
        tokenized_caption = sample['caption']
        mask = sample['mask'][0]
        
        # Decode the tokenized caption to get the original class name
        original_caption = tokenizer(tokenized_caption, encode=False, mask=mask)[0]
        
        # Convert the tensor image back to a PIL image for visualization
        image = T.ToPILImage()(image)
        
        # Display the image
        plt.figure(figsize=(4, 4))
        plt.imshow(image,cmap="gray")
        plt.title(f"Class: {original_caption}")  # Display the non-tokenized class name as title
        plt.axis('off')
        plt.show()

# Visualize samples from the train dataset
# visualize_samples(train_dataset, tokenizer, num_samples=5)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")


    model = CLIP(emb_dim, vit_layers, vit_d_model, img_size,patch_size,n_channels, vit_heads, vocab_size, max_seq_length, text_heads, text_layers,text_d_model, retrieval = False).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    total_params = 0

    total_params = sum([ param.numel() for param in model.parameters() if param.requires_grad])

    print(f"Total number of trainable parameters: {total_params}; i.e., {total_params/1000000:.2f} M")

    best_loss = np.inf
    for epoch in range(epochs):
        epoch_loss = 0.0  # To accumulate the loss over the epoch
        with tqdm(enumerate(train_loader, 0), total=len(train_loader), desc=f"Epoch [{epoch+1}/{epochs}]") as tepoch:
            for i, data in tepoch:
                img, cap, mask = data["image"].to(device), data["caption"].to(device), data["mask"].to(device)
                # print(mask)
                optimizer.zero_grad()
                loss = model(img, cap, mask)
                loss.backward()
                optimizer.step()

                # Update the progress bar with the current loss
                tepoch.set_postfix(loss=loss.item())
                epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.3f}")

        # Save model if it performed better than the previous best
        if avg_loss <= best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "clip.pt")
            print("Model Saved.")
            
if __name__ == '__main__':
    # freeze_support()
    main()