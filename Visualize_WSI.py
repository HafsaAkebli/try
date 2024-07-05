#Graph construction and GNN framework
import os
import matplotlib.pyplot as plt
import torch
#deal with slides
from openslide import open_slide
from glob import glob
import numpy as np


# define the device (GPU, else CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())

#so in each subfolder of these ones we have like Subset1_Train_1 containing tiff images : G3_Mask.tif, G4_Mask.tif, Normal_Mask.tif,
#Stroma_Mask.tif and sometimes also G5_Mask.tif sometimes just 3 types, sometimes the 5 types

#path_masks_train="/mnt/dmif-nas/MITEL/challenges/AGGC22/ProMaL/resized-masks/train/"
#path_masks_valid= "/mnt/dmif-nas/MITEL/challenges/AGGC22/ProMaL/resized-masks/valid/"
slides= "/mnt/dmif-nas/MITEL/challenges/AGGC22/ProMaL/slides/"

# Function to list all .tif files in subdirectories
def list_tif_files(main_dir):
    return (glob(os.path.join(main_dir, '**', '*.tiff'), recursive=True))

print(list_tif_files(slides))
print(len(list_tif_files(slides)))


# visualize an image using OpenSlide and matplotlib
def visualize_image(file_path, save_dir):
    # Open the whole slide image
    slide = open_slide(file_path)
    
    # Read the whole slide image at the highest resolution (level 0)
    img = np.array(slide.read_region((0, 0), 0, slide.level_dimensions[0]))
    
    # Display the image using matplotlib
    plt.imshow(img)
    plt.title(os.path.basename(file_path))
    plt.axis('off')
    
    # Save the image to the specified directory
    file_name = os.path.basename(file_path).replace('.tiff', '.png')
    save_path = os.path.join(save_dir, file_name)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    
    # Close the slide object
    slide.close()
    print(f"Saved image to {save_path}")

visualize_image("/mnt/dmif-nas/MITEL/challenges/AGGC22/ProMaL/slides/Subset2_Train_34.tiff","/home/akebli/test5/try/")


#print(len(path_masks_train))

#print(len(path_masks_valid))