import tifffile
import numpy as np
import os

def evaltile(im, cat):
    for j in range(0, y - tilesize, stride):
        for i in range(0, x - tilesize, stride):
            tile = image[j:j + tilesize, i:i + tilesize]
            percent = np.mean(tile)
            if percent > mincontent:
                outstring=im+"\t"+cat+"\t"+str(i*10)+\
				"\t"+str(j*10)+"\t"+str(percent)+"\n"
                fh.write(outstring)

# Parameters
stride = 10  # Ensured proper sliding window movement
mincontent = 0.7  # Minimum tissue content threshold for tiles

maindir = "/mnt/dmif-nas/MITEL/challenges/AGGC22/ProMaL/resized-masks/train/"
outfile = "/home/akebli/test5/try/tiles_list_subset2.txt"

# Create output directory if it doesn't exist
os.makedirs(os.path.dirname(outfile), exist_ok=True)

with open(outfile, "w") as fh:
    imlist = os.listdir(maindir)
    tilesize = 50  # Because the mask is downscaled by 10 from the WSI
    tileradius = tilesize // 2
    subset = "Subset2"

    for img in imlist:
        if img.startswith(subset):
            masklist = os.listdir(os.path.join(maindir, img))
            for mask in masklist:
                if mask.endswith(".tif"):
                    image = tifffile.imread(os.path.join(maindir, img, mask))
                    y, x = image.shape
                    im = img
                    cat = mask.split('_')[0]  # Extract the class name from the name of the mask
                    evaltile(im, cat)