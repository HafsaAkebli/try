from PIL import Image
import os
import numpy as np
from histoencoder import Histoencoder

# Paths
patch_folder = "/home/akebli/test5/patches/"

# Initialize Histoencoder (assumed library or method)
histoencoder = Histoencoder()

# Load patches
patches = []
for cls in ["G3", "G4", "G5", "Stroma", "Normal"]:
    cls_folder = os.path.join(patch_folder, cls)
    for patch_file in os.listdir(cls_folder):
        if patch_file.endswith(".jpg"):
            patch_path = os.path.join(cls_folder, patch_file)
            patch_image = Image.open(patch_path)
            patches.append(patch_image)

# Extract features from patches
features = []
for patch in patches:
    # Convert patch to numpy array
    patch_np = np.array(patch)
    
    # Extract features using Histoencoder (example method)
    feature_vector = histoencoder.extract_features(patch_np)
    
    # Append feature vector to list
    features.append(feature_vector)

# Now 'features' contains feature vectors for each patch
print(f"Extracted features for {len(features)} patches.")
