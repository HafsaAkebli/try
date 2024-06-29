import os
import numpy as np
from PIL import Image
import timm
from histoencoder.functional import create_encoder, extract_features
import torch
from torchvision import transforms


# Define parameters
tilesize = 500  # Size of the tile (500x500 pixels)
tileradius = tilesize // 2  # Half the size of the tile (250 pixels)
step = 250  # Step for the sliding window

# Load the pretrained model
def load_pretrained_model(model_name: str):
    encoder = timm.create_model(model_name, pretrained=True)
    encoder.eval()  # Set the model to evaluation mode
     # Define preprocessing steps
    preprocess = transforms.Compose([
        transforms.Resize((384, 384)),  # Resize to 384x384
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return encoder, preprocess

# Define the model name
model_name = 'xcit_large_24_p8_384_dist'
patch_folder = "/home/akebli/test5/patches/"
output_features_file = "/home/akebli/test5/features_xcit_large_24_p8_384_dist.npz"

# Load the pretrained model and preprocessing transformations
encoder, preprocess = load_pretrained_model(model_name)

def extract_features_from_patches(encoder, preprocess, patch_folder: str):
    features = []
    labels = []
    for cls in ["G3", "G4", "G5", "Stroma", "Normal"]:
        cls_folder = os.path.join(patch_folder, cls)
        for patch_file in os.listdir(cls_folder):
            if patch_file.endswith(".jpg"):
                patch_path = os.path.join(cls_folder, patch_file)
                patch_image = Image.open(patch_path).convert("RGB")

                # Apply transformations
                patch_image = preprocess(patch_image)

                # Convert to a PyTorch tensor and add batch dimension
                patch_tensor = patch_image.unsqueeze(0)

                # Extract features
                with torch.no_grad():
                    feature_vector = extract_features(
                        encoder, patch_tensor, num_blocks=1, avg_pool=False
                    ).cpu().numpy().flatten()  # Get features from the model

                # Append feature vector and label to lists
                features.append(feature_vector)
                labels.append(cls)
    return np.array(features), np.array(labels)

features, labels = extract_features_from_patches(encoder, preprocess, patch_folder)

def save_features_to_file(features: np.ndarray, labels: np.ndarray, output_file: str):
    np.savez(output_file, features=features, labels=labels)

output_features_file = "/home/akebli/test5/features_xcit_large_24_p8_384_dist.npz"
# Save features and labels to a file
save_features_to_file(features, labels, output_features_file)

print(f"Features and labels have been saved to {output_features_file}")