import numpy as np
import os
from collections import defaultdict
from PIL import Image
import networkx as nx
import matplotlib.pyplot as plt
import openslide  # For handling large WSI images
from sklearn.neighbors import NearestNeighbors  # For KNN searches

# Disable the decompression bomb check for large images
Image.MAX_IMAGE_PIXELS = None

# Define your output features file
output_features_file = "/home/akebli/test5/features_prostate_medium.npz"

# Load the .npz file containing features, labels, and patch paths
data = np.load(output_features_file)
features = data['features']
labels = data['labels']
patch_paths = data['patch_paths']

# Create a mapping from patch path to feature vector
patch_to_feature = dict(zip(patch_paths, features))

# Function to extract WSI name and coordinates from patch paths
def extract_name_wsi_and_coords(filename):
    """Extract WSI ID and coordinates from the patch filename."""
    parts = filename.split('-')
    if len(parts) == 3:
        wsi_id = parts[0]
        x = int(parts[1])
        y = int(parts[2].split('.')[0])  # Remove the extension
    elif len(parts) == 4:
        wsi_id = '-'.join(parts[:3])
        x = int(parts[3])
        y = int(parts[4].split('.')[0])  # Remove the extension
    else:
        raise ValueError(f"Unexpected filename format: {filename}")
    return wsi_id, x, y

# Group patches by WSI and extract coordinates
def organize_patches_by_wsi(patch_paths):
    """Organize patches into a dictionary by WSI ID and extract patch coordinates."""
    wsi_patches = defaultdict(lambda: {'paths': [], 'coords': []})
    for patch_path in patch_paths:
        try:
            wsi_name, x, y = extract_name_wsi_and_coords(os.path.basename(patch_path))
            wsi_patches[wsi_name]['paths'].append(patch_path)
            wsi_patches[wsi_name]['coords'].append((x, y))
        except Exception as e:
            print(f"Skipping patch due to error: {e}")
    return wsi_patches

# Organize patches by WSI
wsi_patches = organize_patches_by_wsi(patch_paths)
print(f"Patches are grouped by WSI: {len(wsi_patches)} WSIs found.")

# Build graph for each WSI
def build_graph_for_wsi(wsi_patches, k=5):
    """Build a KNN graph for each WSI where edges represent nearest neighbors."""
    graphs = {}
    for wsi_id, data in wsi_patches.items():
        patches = data['paths']
        coords = data['coords']
        patch_features = np.array([patch_to_feature[patch] for patch in patches])

        # Initialize NearestNeighbors and fit the model
        nn = NearestNeighbors(n_neighbors=k + 1, metric='euclidean')
        nn.fit(patch_features)
        distances, indices = nn.kneighbors(patch_features)

        G = nx.Graph()
        for i, patch_path in enumerate(patches):
            G.add_node(patch_path, feature=patch_to_feature[patch_path], pos=coords[i])
            for j in range(1, k + 1):  # Skip the 0-th neighbor which is the node itself
                neighbor_idx = indices[i, j]
                neighbor_patch = patches[neighbor_idx]
                distance = distances[i, j]
                G.add_edge(patch_path, neighbor_patch, weight=distance)
        
        graphs[wsi_id] = G
    return graphs

# Build graphs for each WSI
graphs = build_graph_for_wsi(wsi_patches)
print(f"Graphs have been built for {len(graphs)} WSIs.")

# Visualize one WSI graph
def visualize_graph(name_wsi, graph, wsi_image_path=None):
    """Visualize the KNN graph on top of the WSI image."""
    if wsi_image_path:
        # Open the whole slide image using openslide
        slide = openslide.OpenSlide(wsi_image_path)

        # Get the full resolution of the image
        slide_dim = slide.dimensions

        # Create a matplotlib figure
        plt.figure(figsize=(12, 12))

        # Get a thumbnail of the image
        thumbnail_size = (int(slide_dim[0] * 0.1), int(slide_dim[1] * 0.1))  # Example downsample size (10% of original size)
        wsi_image = slide.read_region((0, 0), 0, thumbnail_size)
        wsi_image = wsi_image.convert('RGB')  # Convert to RGB mode for visualization

        # Draw the graph on top of the WSI image
        pos = nx.get_node_attributes(graph, 'pos')

        # Convert positions to match the WSI image coordinates
        pos = {k: (v[0] * 0.1, v[1] * 0.1) for k, v in pos.items()}  # Scale down coordinates by 10%

        plt.imshow(wsi_image, alpha=0.8)  # Use default colormap for H&E images
        nx.draw(
            graph, 
            pos, 
            node_size=5,  # Size of the nodes
            node_color='black',  # Color of the nodes
            edge_color='cyan',  # Color of the edges
            alpha=0.7,  # Transparency of the graph
            width=0.5,  # Width of the edges
            with_labels=False,  # Do not show the labels
            ax=plt.gca()  # Draw on the current axes
        )

        plt.title(f"Graph for WSI: {name_wsi}")

        # Save the figure
        figure_save_path = f"/home/akebli/test5/try/graph_{name_wsi}.png"
        plt.savefig(figure_save_path, bbox_inches='tight')  # Save with tight bounding box
        plt.show()
        print(f"Graph for WSI {name_wsi} saved to {figure_save_path}")

# Select a WSI ID to visualize
wsi_name_to_visualize = 'Subset1_Train_49'
print(f"Visualizing graph for WSI: {wsi_name_to_visualize}")

# Path to the WSI image file
wsi_image_path = f"/mnt/dmif-nas/MITEL/challenges/AGGC22/ProMaL/slides/{wsi_name_to_visualize}.tiff"
visualize_graph(wsi_name_to_visualize, graphs[wsi_name_to_visualize], wsi_image_path=wsi_image_path)

print("done")
