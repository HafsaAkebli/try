import tifffile
import numpy as np
import os
from PIL import Image
import csv
import openslide

#extract and record information about specific tiles within binary mask images based on a predefined threshold (mincontent)

#def evaltile(im, clas):
 #  for j in range(50, y - 50, step):
   #     for i in range(50, x - 50, step):
   #         tile = image[j - tileradius:j + tileradius, i - tileradius:i + tileradius]
    #        percent = np.mean(tile)
    #        if percent > mincontent:
    #            #outstring = f"{im}\t{clas}\t{i}\t{j}\t{percent}\n"
      #          outstring=im+"\t"+clas+"\t"+str(i)+\
	#			"\t"+str(j)+"\t"+str(percent)+"\n"
     #           fh.write(outstring)

#Masks to list Based on the resized Masks 

#for img in imlist:
    #if img.startswith(subset_prefix):
      #  masklist = os.listdir(maindir + img)
       # for mask in masklist:
        #    if mask.endswith(".tif"):
          #      image = tifffile.imread(maindir + img + "/" + mask)
            #    y, x = image.shape
             #   im = img
              #  clas = mask.split('_')[0]  #takes the name of the class of the mask (G3,G4,G5, Stroma,Normal)
              #  mask_name = mask  # Keep the mask name for reference 
              #  evaltile(im, clas)

#fh.close()

# Parameters
tilesize = 500  # Size of the tile (500x500 pixels)
tileradius = tilesize // 2  # Half the size of the tile (250 pixels)
mincontent = 0.9  # Minimum content threshold for tiles
step = 250  # Step for the sliding window

# Input and output paths
maindir = "/mnt/dmif-nas/MITEL/challenges/AGGC22/ProMaL/resized-masks/train/"
outfile = "/home/akebli/test5/tiles_list1.txt"  # Path to output file for the tile list

# Create output directory if it doesn't exist
os.makedirs(os.path.dirname(outfile), exist_ok=True)

fh = open(outfile, "w")
imlist = os.listdir(maindir)
subset_prefix = "Subset"  # Prefix to identify relevant subdirectories

def evaltile(im, clas):
    for j in range(tileradius, y - tileradius, step):
        for i in range(tileradius, x - tileradius, step):
            tile = image[j - tileradius:j + tileradius, i - tileradius:i + tileradius]
            percent = np.mean(tile)
            if percent > mincontent:
                outstring = f"{im}\t{clas}\t{i}\t{j}\t{percent}\n"
                fh.write(outstring)
                

for img in imlist:
    if img.startswith(subset_prefix):
        masklist = os.listdir(maindir + img)
        for mask in masklist:
            if mask.endswith(".tif"):
                image = tifffile.imread(maindir + img + "/" + mask)
                y, x = image.shape
                im = img
                clas = mask.split('_')[0]  # Takes the name of the class of the mask (G3, G4, G5, Stroma, Normal)
                evaltile(im, clas)

fh.close()