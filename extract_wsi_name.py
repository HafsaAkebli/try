
#Graph construction and GNN framework
import os
import matplotlib.pyplot as plt
import torch
#deal with slides
from openslide import open_slide
from glob import glob
import numpy as np

def extract_name_wsi_and_coords(filename):
    """Extract WSI name and coordinates of the patch from the patch filename."""
    parts = filename.split('-')
    wsi_id = "-".join(parts[:-2])
    x = int(parts[-2])
    y = int(parts[-1].split('.')[0])  # Remove the extension
    return wsi_id, x, y

print (extract_name_wsi_and_coords("Subset3_Train_8-6500-58750.jpg"))