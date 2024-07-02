import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import histoencoder.functional as F
from histoencoder.functional import extract_features

# Define parameters
tilesize = 500  # Size of the tile (500x500 pixels)
tileradius = tilesize // 2  # Half the size of the tile (250 pixels)
step = 250  # Step for the sliding window

# Define the device to use (GPU if available, otherwise CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("CUDA is available:", torch.cuda.is_available())

# Load the pretrained HistoEncoder model
def load_histoencoder_model(model_name: str):
    encoder = F.create_encoder(model_name)
    encoder.eval()  # Set the model to evaluation mode
    encoder.to(device)  # Move the model to the selected device
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to the input size expected by the model
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return encoder, preprocess

# Define the model name
model_name = 'prostate_medium'  # the pretrained model of histoencoder, there is also prostate_small
patch_folder = "/home/akebli/test5/patches/"
output_features_file = "/home/akebli/test5/features_prostate_medium.npz"
model_save_path = "/home/akebli/test5/prostate_medium_model.pth"  # Path to save the model

# Load the pretrained model and preprocessing transformations
encoder, preprocess = load_histoencoder_model(model_name)
print("Encoder loaded")

# Save the model
torch.save(encoder.state_dict(), model_save_path)
print(f"Model has been saved to {model_save_path}")

# Extract features from patches
def extract_features_from_patches(encoder, preprocess, patch_folder: str):
    features = []
    labels = []
    for cls in ["G3", "G4", "G5", "Stroma", "Normal"]:
        cls_folder = os.path.join(patch_folder, cls)
        for patch_file in os.listdir(cls_folder):
            if patch_file.endswith(".jpg"):
                patch_path = os.path.join(cls_folder, patch_file)
                patch_image = Image.open(patch_path)

                # Apply transformations
                patch_image = preprocess(patch_image)

                # Convert to a PyTorch tensor and add batch dimension
                patch_tensor = patch_image.unsqueeze(0).to(device)  # Move the tensor to the device

                # Extract features
                with torch.no_grad():
                    feature_vector = F.extract_features(
                        encoder, patch_tensor, num_blocks=1, avg_pool=False
                    ).cpu().numpy().flatten()  # Get features from the model and move to CPU

                # Append feature vector and label to lists
                features.append(feature_vector)
                labels.append(cls)
    return np.array(features), np.array(labels)

# Extract features and labels
features, labels = extract_features_from_patches(encoder, preprocess, patch_folder)

# Save features to a file
def save_features_to_file(features: np.ndarray, labels: np.ndarray, output_file: str):
    np.savez(output_file, features=features, labels=labels)

# Save features and labels to a file
save_features_to_file(features, labels, output_features_file)

print(f"Features and labels have been saved to {output_features_file}")
