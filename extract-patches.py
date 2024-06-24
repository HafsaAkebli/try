import numpy as np
import os
from PIL import Image
import csv
import openslide
import tifffile

# Define parameters
tilesize = 256  # Size of the tile (256x256 pixels)
tileradius = tilesize // 2  # Half the size of the tile (128 pixels)
mincontent = 0.9  # Minimum content threshold for tiles
step = 128  # Step for the sliding window

# Input and output paths
maindir = "/mnt/dmif-nas/MITEL/challenges/AGGC22/ProMaL/resized-masks/train/"
tiles_list_file = "/home/akebli/test5/tiles_list1.txt"
source_folder = "/mnt/dmif-nas/MITEL/challenges/AGGC22/ProMaL/slides"
destination_folder = "/home/akebli/test5/patches/"



for cls in ["G3", "G4", "G5", "Stroma", "Normal"]:  # classes
    os.makedirs(os.path.join(destination_folder, cls), exist_ok=True)

# Read tiles list
tiles = []
with open(tiles_list_file, 'r') as tilefile:
    csv_reader = csv.reader(tilefile, delimiter='\t')
    for row in csv_reader:
        tiles.append(row)

# Process each tile
for t in tiles:
    img_name = t[0]
    cls = t[1]
    x = int(t[2]) // 10  # Convert to original coordinates
    y = int(t[3]) // 10  # Convert to original coordinates

    # Construct the path to the WSI
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
    x_start = max(0, x - tileradius)
    y_start = max(0, y - tileradius)
    x_end = min(img.dimensions[0], x + tileradius)
    y_end = min(img.dimensions[1], y + tileradius)

    # Extract tile from WSI
    tile_data = img.read_region((x_start, y_start), 0, (tilesize, tilesize))
    tile_data_np = np.array(tile_data)

    # Calculate percentage of tissue
    tile_gray = tile_data_np[:, :, 0]  # Assuming grayscale
    percent_tissue = np.mean(tile_gray) / 255.0  # Normalize to 0-1

    # Check if percentage meets threshold
    if percent_tissue > mincontent:
        # Save the tile to destination folder
        tile_filename = f"{img_name}-{x}-{y}.jpg"
        tile_path = os.path.join(destination_folder, cls, tile_filename)
        im = Image.fromarray(tile_data_np)
        im = im.convert("RGB")
        im.save(tile_path)

    # Close the WSI
    img.close()
