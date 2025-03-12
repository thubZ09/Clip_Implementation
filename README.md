# CLIP Implementation 

This repository comtains an implementation of a CLIP-like model that aligns image and text representations using PyTorch and Hugging Face Transformers. The model uses a Vision Transformer (ViT) as the image encoder and DistilRoBERTa as the text encoder. It is trained on the Flickr30k dataset.

- Paper - https://arxiv.org/pdf/2103.00020

## 📌Table of Contents

- [Overview](#overview)
- [Working](#working)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Data Setup](#data-setup)
- [Inference](#inference)
- [Notes](#notes)


## 📌Overview

This project demonstrates how to implement a CLIP-like architecture locally using your preferred IDE. The model is designed to learn joint representations for images and their corresponding captions, which can then be used for tasks such as image retrieval, caption generation, and cross-modal matching.

## 📌Working

- **Image Encoder:** Uses `google/vit-base-patch16-224-in21k`.
- **Text Encoder:** Uses `distilroberta-base`.
- **Training Script:** Includes a training loop with support for gradient scaling and learning rate scheduling.
- **Inference Script:** Provides a utility to find the top matching captions for a given image.
- **Local Development:** Configured to run locally with adjustable paths and settings.

## 📌Structure

```plaintext
CLIP_Implementation/
├── checkpoints/                   # Folder to save model checkpoints
├── data/                          # Contains dataset files
│   └── flickr-image-dataset/      # Unzipped Flickr30k dataset folder (images and results.csv)
├── __init__.py                    # (empty file, marks package)
├── configs.py                     # Configuration settings (models, training parameters, etc.)
├── data_utils.py                  # Data loading and preprocessing utilities
├── model.py                       # CLIP model definition
├── train.py                       # Training script
└── inference.py                   # Inference utilities for testing the model
```
## 📌Requirements

- Python 3.7 or higher
- PyTorch
- Transformers
- Torchvision
- Pandas
- Pillow
- Kaggle API (optional)

## 📌Installation
```plaintext
CLONE THE REPO:
git clone https://github.com/thubZ09/Clip_Implementation.git
cd Clip_Implementation

CREATE A VIRTUAL ENV:
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

INSTALL DEPENDENCIES:
pip install torch torchvision transformers pandas Pillow kaggle
```

## 📌Data Setup  
- Download the Dataset -
 Flickr30k dataset. You can use the Kaggle API or manually download and unzip the dataset.

- Place the Dataset -
Ensure the unzipped dataset is located in the ./data/flickr-image-dataset/ folder. Verify that the folder contains both the results.csv file and the image files.

## 📌Inference

To test the model and retrieve the top matching captions for an image, use the find_matches function in inference.py. For example:
```plaintext
from model import LocalCLIP
from inference import find_matches
import torch

# Initialize your model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = LocalCLIP().to(device)

# Example usage: Retrieve top 5 captions for a given image
top_matches = find_matches(model, "path/to/image.jpg", ["caption one", "caption two", "caption three"], device=device)
print(top_matches)
```

## 📌Notes  
- Performance -
Training on a CPU may be very slow. Adjust batch size and other parameters as needed, or use a GPU if available.

- Environment -
It is designed for local development using your preferred IDE. Make sure to configure your virtual environment and paths correctly.

