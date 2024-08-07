import numpy as np

# Path to the .npz file
output_features_file = "/home/akebli/test5/features_Subset1_valid_prostate_medium.npz"

# Load the .npz file
data = np.load(output_features_file)

# Access the features and labels
features = data['features']
labels = data['labels']
patch_paths = data['patch_paths']

# Check the shapes of the loaded arrays
print("Features shape:", features.shape)
print("Labels shape:", labels.shape)
print("patch_paths" , patch_paths.shape)

# Display the first few entries of the features and labels
print("First 10 feature vectors:\n", features[10:])
print("First 10 labels:\n", labels[10:])
print("First 10 patch paths:\n", patch_paths[10:])
