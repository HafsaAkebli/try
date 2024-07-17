import numpy as np
import os
import csv
import openslide
from PIL import Image
import tifffile

# Define parameters
patch_size = 500  # Size of the patch (500x500 pixels)
tile_radius = patch_size // 2  # Half the size of the patch (250 pixels)
stride = 250  # Step for the sliding window

# Input and output paths
maindir = "/mnt/dmif-nas/MITEL/challenges/AGGC22/ProMaL/resized-masks/valid/"
tiles_list_file = "/home/akebli/test5/try/tiles_list_valid.txt"
source_folder = "/mnt/dmif-nas/MITEL/challenges/AGGC22/ProMaL/slides"
destination_folder = "/home/akebli/test5/Patches1/"

# Create directories for each class if they don't exist
for cls in ["G3", "G4", "G5", "Stroma", "Normal"]:
    os.makedirs(os.path.join(destination_folder, cls), exist_ok=True)

# Read the list of tiles (text file), store the rows in a list
tiles = []
with open(tiles_list_file, 'r') as tilefile:
    csv_reader = csv.reader(tilefile, delimiter='\t')
    for row in csv_reader:
        tiles.append(row)

# Process each tile
for t in tiles:
    img_name = t[0]
    cls = t[1]  # Class of the patch
    x = int(t[2])  # X-coordinate of the patch
    y = int(t[3])  # Y-coordinate of the patch
    subset = img_name.split('_')[0]  # Extract subset from the image name

    # Construct the path to the WSI file
    wsi_path = os.path.join(source_folder, img_name + ".tiff")

    if not os.path.isfile(wsi_path):
        print(f"WSI file not found: {wsi_path}")
        continue

    # Open the WSI
    try:
        img = openslide.OpenSlide(wsi_path)
    except Exception as e:
        print(f"Error opening WSI {wsi_path}: {e}")
        continue

    # Calculate coordinates for patch extraction
    x_start = max(0, x - tile_radius)
    x_end = min(img.dimensions[0], x + tile_radius)

    y_start = max(0, y - tile_radius)
    y_end = min(img.dimensions[1], y + tile_radius)

    # Extract tile from WSI
    tile_data = img.read_region((x_start, y_start), 0, (patch_size, patch_size))
    tile_data_np = np.array(tile_data)

    # Convert to RGB and resize the tile
    tile_image = Image.fromarray(tile_data_np)
    tile_image = tile_image.convert("RGB")
    tile_image_resized = tile_image.resize((224, 224))

    # Create the directory for the subset and class if it doesn't exist
    subset_class_dir = os.path.join(destination_folder, subset, "valid", cls)
    os.makedirs(subset_class_dir, exist_ok=True)

    # Save the resized tile
    tile_filename = f"{img_name}-{x}-{y}.jpg"
    tile_path = os.path.join(subset_class_dir, tile_filename)
    tile_image_resized.save(tile_path)

    # Close the WSI
    img.close()
