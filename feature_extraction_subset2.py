import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import histoencoder.functional as F
from histoencoder.functional import extract_features

# Define parameters
tilesize = 500  # Size of the tile (500x500 pixels)
tileradius = tilesize // 2  # Half the size of the tile (250 pixels)
step = 250  # Step for the sliding window
batch_size = 32  # Define your batch size here

# Define the device to use (GPU if available, otherwise CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("CUDA is available:", torch.cuda.is_available())

# Define a custom dataset for loading patches
class PatchDataset(Dataset):
    def __init__(self, patch_folder, transform=None):
        self.patch_folder = patch_folder
        self.transform = transform
        self.classes = ["G3", "G4", "G5", "Stroma", "Normal"]
        self.filepaths = []
        self.labels = []
        
        for cls in self.classes:
            cls_folder = os.path.join(self.patch_folder, cls)
            for patch_file in os.listdir(cls_folder):
                if patch_file.endswith(".jpg"):
                    self.filepaths.append(os.path.join(cls_folder, patch_file))
                    self.labels.append(cls)
                    
    def __len__(self):
        return len(self.filepaths)
    
    def __getitem__(self, idx):
        patch_path = self.filepaths[idx]
        label = self.labels[idx]
        patch_image = Image.open(patch_path)
        
        if self.transform:
            patch_image = self.transform(patch_image)
        
        return patch_image, label, patch_path

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

# Define the patch folders for Subset1 and Subset3
patch_folders = ["/home/akebli/test5/Patches1/Subset2/train"]

# Load the pretrained model and preprocessing transformations
encoder, preprocess = load_histoencoder_model(model_name)
print("Encoder loaded successfully")

# Save the model (if needed)
model_path = "/home/akebli/test5/histoencoder_prostate_medium.pth"
torch.save(encoder.state_dict(), model_path)
print(f"Model has been saved to {model_path}")

# Extract features from patches in batches
def extract_features_from_batches(encoder, dataloader):
    features = []
    labels = []
    patch_paths = []
    for images, batch_labels, paths in dataloader:
        images = images.to(device)
        with torch.no_grad():
            batch_features = F.extract_features(encoder, images, num_blocks=1, avg_pool=False)
        features.append(batch_features.cpu().numpy())
        labels.extend(batch_labels)
        patch_paths.extend(paths)
    return np.concatenate(features, axis=0), np.array(labels), np.array(patch_paths)

# Iterate over the patch folders for Subset1 and Subset3
for patch_folder in patch_folders:
    subset_name = os.path.basename(os.path.dirname(patch_folder))  # Get the subset name (e.g., Subset1, Subset3)
    output_features_file = f"/home/akebli/test5/features_{subset_name}_1.npz"
    
    # Create Dataset and DataLoader
    dataset = PatchDataset(patch_folder, transform=preprocess)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Extract features, labels, and patch paths
    features, labels, patch_paths = extract_features_from_batches(encoder, dataloader)
    print(f"Features extracted successfully for {subset_name}")
    
    # Save features, labels, and patch paths to a file
    def save_features_to_file(features: np.ndarray, labels: np.ndarray, patch_paths: np.ndarray, output_file: str):
        np.savez(output_file, features=features, labels=labels, patch_paths=patch_paths)
    
    # Save features, labels, and patch paths to a file
    save_features_to_file(features, labels, patch_paths, output_features_file)
    print(f"Features, labels, and patch paths have been saved to {output_features_file}")

print("Feature extraction and saving completed for Subset2.")
