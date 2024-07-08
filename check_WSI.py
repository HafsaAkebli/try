import os
from collections import defaultdict

# Define the path to the patches directory
patches_dir = "/home/akebli/test5/Patches1/Subset1/train"

# Define the categories
categories = ["G3", "G4", "G5", "Stroma", "Normal"]

# Function to extract the WSI name from the patch filename
def extract_wsi_name(patch_filename):
    parts = patch_filename.split('-')
    wsi_name = "-".join(parts[:-2])
    return wsi_name

# Initialize a dictionary to hold the count of patches for each WSI
wsi_patch_count = defaultdict(int)

# Loop through each category directory and count patches for each WSI
for category in categories:
    category_dir = os.path.join(patches_dir, category)
    for patch_filename in os.listdir(category_dir):
        if patch_filename.endswith(".jpg"):
            wsi_name = extract_wsi_name(patch_filename)
            wsi_patch_count[wsi_name] += 1

# Print the number of patches for each WSI
print(f"Total WSIs found: {len(wsi_patch_count)}")
for wsi_name, count in wsi_patch_count.items():
    print(f"{wsi_name}: {count} patches")

