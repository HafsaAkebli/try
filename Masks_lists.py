import tifffile
import numpy as np
import os
from PIL import Image


#extract and record information about specific tiles within binary mask images based on a predefined threshold (mincontent)

def evaltile(im, clas):
    for j in range(50, y - 50, step):
        for i in range(50, x - 50, step):
            tile = image[j - tileradius:j + tileradius, i - tileradius:i + tileradius]
            percent = np.mean(tile)
            if percent > mincontent:
                #outstring = f"{im}\t{clas}\t{i}\t{j}\t{percent}\n"
                outstring=im+"\t"+clas+"\t"+str(i*10)+\
				"\t"+str(j*10)+"\t"+str(percent)+"\n"
                fh.write(outstring)

#Masks to list Based on the resized Masks 

# Define paths and parameters directly


maindir = "/mnt/dmif-nas/MITEL/challenges/AGGC22/ProMaL/resized-masks/train/"  # Path to masks directory
outfile = "/home/akebli/test5/tiles_list1.txt"  # Path to output file

# Create output directory if it doesn't exist
os.makedirs(os.path.dirname(outfile), exist_ok=True)

fh = open(outfile, "w")
imlist = os.listdir(maindir)

mincontent = 0.9  # Minimum content threshold for tiles
tilesize = 256  # Size of the tile (256x256 pixels)
tileradius = tilesize // 2  # Half the size of the tile (10 pixels)
subset_prefix = "Subset"  # Prefix to identify relevant subdirectories
step = 128  #step for the sliding window

for img in imlist:
    if img.startswith(subset_prefix):
        masklist = os.listdir(maindir + img)
        for mask in masklist:
            if mask.endswith(".tif"):
                image = tifffile.imread(maindir + img + "/" + mask)
                y, x = image.shape
                im = img
                clas = mask.split('_')[0]  #takes the name of the class of the mask (G3,G4,G5, Stroma,Normal)
                mask_name = mask  # Keep the mask name for reference 
                evaltile(im, clas)

fh.close()
