import tifffile
import numpy as np
import os

def evaltile(im, cat):
    for j in range(0, y - tilesize, stride):
        for i in range(0, x - tilesize, stride):
            tile = image[j:j + tilesize]
            percent = np.mean(tile)
            if percent > mincontent:
                outstring=im+"\t"+cat+"\t"+str(i*10)+\
				"\t"+str(j*10)+"\t"+str(percent)+"\n"
                fh.write(outstring)

# Parameters
stride = 250//10 #Ensured proper sliding window movement.
maindir = "/mnt/dmif-nas/MITEL/challenges/AGGC22/ProMaL/resized-masks/train/"
outfile = "/home/akebli/test5/try/tiles_list1.txt"

# Create output file
os.makedirs(os.path.dirname(outfile), exist_ok=True)
fh = open(outfile, "w")

imlist = os.listdir(maindir)
mincontent = 0.95  # Minimum content threshold for tiles
tilesize = 50  # Cause the mask is downscaled by 10 from the WSI
tileradius = tilesize // 2
subset = "Subset"

for img in imlist:
    if img.startswith(subset):
        masklist = os.listdir(maindir + img)
        for mask in masklist:
            if mask.endswith(".tif"):
                image = tifffile.imread(maindir + img + "/" + mask)
                y, x = image.shape
                im = img
                cat = mask.split('_')[0]  # Extract the class name from the name of the mask
                evaltile(im, cat)

fh.close()