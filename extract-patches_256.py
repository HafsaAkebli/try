import numpy as np
import os
from PIL import Image
import csv
import openslide
import tifffile

# Define parameters
tilesize = 500  # Size of the tile (256x256 pixels)
tileradius = tilesize // 2  # Half the size of the tile (128 pixels)
mincontent = 0.95  # Minimum content threshold for tiles
step = 250  # Step for the sliding window

# Input and output paths
maindir = "/mnt/dmif-nas/MITEL/challenges/AGGC22/ProMaL/resized-masks/train/"
tiles_list_file = "/home/akebli/test5/try/tiles_list1.txt"
source_folder = "/mnt/dmif-nas/MITEL/challenges/AGGC22/ProMaL/slides"  #the path of the original slides
destination_folder = "/home/akebli/test5/patches/"


#Create the directories for each patch corresponding to each class to be stored in the folders created
for cls in ["G3", "G4", "G5", "Stroma", "Normal"]: 
    os.makedirs(os.path.join(destination_folder, cls), exist_ok=True)

# Read the list of tiles (text file), store teh rows in a list
tiles = []
with open(tiles_list_file, 'r') as tilefile:
    csv_reader = csv.reader(tilefile, delimiter='\t')
    for row in csv_reader:
        tiles.append(row)

# Process each tile
for t in tiles:
    img_name = t[0]
    cls = t[1]  #class of the patch
    x = int(t[2])  # Convert to original coordinates
    y = int(t[3])  # Convert to original coordinates

    # Construct the path to the WSI according to the name of the patch
    wsi_path = os.path.join(source_folder, img_name + ".tiff")
    
    if not os.path.isfile(wsi_path):
        print(f"WSI file not found: {wsi_path}")
        continue

    # Open the WSI having the path that we just construct
    try:
        img = openslide.OpenSlide(wsi_path)
    except Exception as e:
        print(f"Error opening WSI {wsi_path}: {e}")
        continue

    # Calculate coordinates for patch extraction
    x_start = max(0, x - tileradius)
    x_end = min(img.dimensions[0], x + tileradius)

    y_start = max(0, y - tileradius)
    y_end = min(img.dimensions[1], y + tileradius)

    # Extract tile from WSI
    tile_data = img.read_region((x_start, y_start), 0, (tilesize, tilesize))
    tile_data_np = np.array(tile_data)

   
    # Convert the tile to RGB and save it in the destination folder
    tile_filename = f"{img_name}-{x}-{y}.jpg"
    tile_path = os.path.join(destination_folder, cls, tile_filename)
    im = Image.fromarray(tile_data_np)
    im = im.convert("RGB")
    im=im.resize((224,224))
    im.save(tile_path)

    # Close the WSI
    img.close()
