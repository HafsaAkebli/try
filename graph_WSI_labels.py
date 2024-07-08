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

# Define your output features file (replace with the correct subset names)
subset_names = ['Subset1', 'Subset3']
output_features_files = [f"/home/akebli/test5/features_{subset_name}_train_prostate_medium.npz" for subset_name in subset_names]

# Function to load features, labels, and patch paths from multiple files
def load_features(files):
    features = []
    labels = []
    patch_paths = []
    for file in files:
        data = np.load(file)
        features.append(data['features'])
        labels.append(data['labels'])
        patch_paths.append(data['patch_paths'])
    return np.concatenate(features), np.concatenate(labels), np.concatenate(patch_paths)

# Load the .npz files containing features, labels, and patch paths
features, labels, patch_paths = load_features(output_features_files)

# Create a mapping from patch path to feature vector
patch_to_feature = dict(zip(patch_paths, features))
# Create a mapping from patch path to the patch label or class
patch_to_label = dict(zip(patch_paths, labels))

# Extract WSI name and coordinates from patch paths
def extract_name_wsi_and_coords(filename):
    """Extract WSI name and coordinates of the patch from the patch filename."""
    parts = filename.split('-')
    wsi_id = "-".join(parts[:-2])  # first parts
    x = int(parts[-2])  # coordinate x
    y = int(parts[-1].split('.')[0])  # coordinate y Remove the extension
    return wsi_id, x, y
#va bene

# Group patches by WSI and extract coordinates

def organize_patches_by_wsi(patch_paths):
    #Organize patches into a dictionary by WSI name and extract patch coordinates and labels and also the paths of these patches"""
    wsi_patches = defaultdict(lambda: {'paths': [], 'coords': [], 'labels': []})
    for patch_path in patch_paths:
        try:
            wsi_name, x, y = extract_name_wsi_and_coords(os.path.basename(patch_path))
            centroid_x = x + 250  # Calculate centroid x-coordinate
            centroid_y = y + 250  # Calculate centroid y-coordinate
            wsi_patches[wsi_name]['paths'].append(patch_path)
            wsi_patches[wsi_name]['coords'].append((centroid_x, centroid_y))
            wsi_patches[wsi_name]['labels'].append(patch_to_label[patch_path])
        except Exception as e:
            print(f"Skipping patch due to error: {e}")
    return wsi_patches

wsi_patches = organize_patches_by_wsi(patch_paths)
print(f"Patches are grouped by WSI: {len(wsi_patches)} WSIs found.")

#Now the patches are organized by WSI name, we can build the graph
# Build graph for each WSI
def build_graph_for_wsi(wsi_patches, k=5):
    """Build a KNN graph for each WSI where edges represent nearest neighbors."""
    graphs = {}
    for wsi_id, data in wsi_patches.items():
        patches = data['paths']
        coords = data['coords']
        labels = data['labels']
        patch_features = np.array([patch_to_feature[patch] for patch in patches])

        # Initialize NearestNeighbors and fit the model
        nn = NearestNeighbors(n_neighbors=k + 1, metric='euclidean')
        nn.fit(patch_features)
        distances, indices = nn.kneighbors(patch_features)

        G = nx.Graph()
        for i, patch_path in enumerate(patches):
            G.add_node(patch_path, feature=patch_to_feature[patch_path], pos=coords[i], label=labels[i])
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

# Define colors for each class
class_colors = {
    'G3': 'pink',
    'G4': 'green',
    'G5': 'red',
    'Stroma': 'yellow',
    'Normal': 'purple'
}

# Visualize one WSI graph
def visualize_graph(name_wsi, graph, wsi_image_path=None):
    """Visualize the KNN graph on top of the WSI image."""
    if wsi_image_path:
        # Open the whole slide image using openslide
        slide = openslide.OpenSlide(wsi_image_path)

        # Get the full resolution of the image
        slide_dim = slide.dimensions

        # Create a matplotlib figure with a fixed size
        plt.figure(figsize=(50, 50))

        # Get the WSI image at full resolution
        wsi_image = slide.read_region((0, 0), 0, slide_dim).convert('RGB')

        # Draw the graph on top of the WSI image
        pos = nx.get_node_attributes(graph, 'pos')

        plt.imshow(wsi_image)  # Use default colormap for H&E images
        node_colors = [class_colors[graph.nodes[n]['label']] for n in graph.nodes]

        nx.draw(
            graph,
            pos,
            node_size=10,  # Size of the nodes
            node_color=node_colors,  # Color of the nodes based on class
            edge_color='blue',  # Color of the edges
            alpha=0.6,  # Transparency of the graph
            width=0.7,  # Width of the edges
            with_labels=False,  # Do not show the labels
            ax=plt.gca()  # Draw on the current axes
        )

        plt.title(f"Graph for WSI: {name_wsi}")

        # Save the figure
        figure_save_path = f"/home/akebli/test5/try/graph_{name_wsi}_1.png"
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
