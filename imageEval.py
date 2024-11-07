import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageDraw
import numpy as np

# Set device to CUDA if available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the DenseNet169 model with updated weights handling
weights = models.densenet169(pretrained=True)  # Load pre-trained DenseNet169 model
in_features = weights.classifier.in_features  # Get input features for the classifier
weights.classifier = nn.Sequential(
    nn.Dropout(p=0.2),  # Set dropout for regularization
    nn.Linear(in_features, 2)  # Assuming binary classification: real vs. deepfake
)

# Move the model to the same device
model = weights.to(device)



def load_model(model, optimizer, checkpoint_path):
    """Load the model and optimizer state from a checkpoint."""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    metrics = checkpoint['metrics']
    return epoch, metrics

# Load the model's state from the saved checkpoint
checkpoint_path = 'model2.pth'  # Update with your actual path
optimizer = torch.optim.Adam(model.parameters())  # Example optimizer; adjust as necessary
epoch, metrics = load_model(model, optimizer, checkpoint_path)

# Set model to evaluation mode
model.eval()

# Define the image transformations to match the model's expected input
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to 224x224 pixels
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet's mean and std
])

def predict_image(image, model, device):
    """
    Predict whether an image is real or deepfake using a pre-trained model.
    
    Args:
    - image_path (str): Path to the image to predict.
    - model (nn.Module): Loaded and trained model for deepfake detection.
    - device (str): Device to perform computations on ('cuda' or 'cpu').
    
    Returns:
    - str: 'Real' or 'Deepfake' based on model prediction.
    """
    # Load and preprocess the image
      # Ensure image is in RGB format
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device
    
    # Make a prediction
    with torch.no_grad():
        output = model(image)  # Get model output
        _, predicted = torch.max(output, 1)  # Get the predicted class index

    # Map the index to class names
    return 'Real' if predicted.item() == 0 else 'Deepfake'



def final_imageOnly(image):
    #st.title(" Analysis")
    
    # Split image into 9x9 grid
    """grid_images = split_image(image)
    
    # Analyze each grid cell
    deepfake_grid = []
    for grid_image in grid_images:
        result = predict_image(grid_image, model, device)
        deepfake_grid.append(result)
    
    # Highlight deepfake sections
    highlighted_image = highlight_deepfake(image.copy(), deepfake_grid)

    st.write(" analysis completed. Highlighting deepfake sections...")
    
    # Display the final highlighted image
    st.image(highlighted_image, caption='Deepfake Sections Highlighted', use_column_width=True)

    # Display grid results (Optional)
    cols = st.columns(9)
    for i, grid_image in enumerate(grid_images):
        col_idx = i % 9
        with cols[col_idx]:
            st.image(grid_image, caption=deepfake_grid[i], use_column_width=True)"""
    result = predict_image(image, model, device)
    #st.write(f" {result}")
    return result 
    
