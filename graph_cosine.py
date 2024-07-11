import numpy as np
import os
from collections import defaultdict
from PIL import Image
import networkx as nx
import matplotlib.pyplot as plt
import openslide
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

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

# Group patches by WSI and extract coordinates
def organize_patches_by_wsi(patch_paths):
    """Organize patches into a dictionary by WSI name and extract patch coordinates and labels and also the paths of these patches"""
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
        nn = NearestNeighbors(n_neighbors=k + 1, metric='cosine')
        nn.fit(patch_features)
        distances, indices = nn.kneighbors(patch_features)

        # Calculate cosine similarities
        similarities = cosine_similarity(patch_features)

        G = nx.Graph()
        for i, patch_path in enumerate(patches):
            G.add_node(patch_path, feature=patch_to_feature[patch_path], pos=coords[i], label=labels[i])
            for j in range(1, k + 1):  # Skip the 0-th neighbor which is the node itself
                neighbor_idx = indices[i, j]
                neighbor_patch = patches[neighbor_idx]
                # Use cosine similarity as the weight (1 - similarity to convert to distance-like weight)
                similarity = similarities[i, neighbor_idx]
                G.add_edge(patch_path, neighbor_patch, weight=(1 - similarity))

        graphs[wsi_id] = G
    return graphs

# Build graphs for each WSI
graphs = build_graph_for_wsi(wsi_patches)
print(f"Graphs have been built for {len(graphs)} WSIs.")

# Define colors for each class
class_colors = {
    'G3': '#FF0000',  # Bright Red
    'G4': '#00FF00',  # Bright Green
    'G5': '#0000FF',  # Bright Blue
    'Stroma': '#FFA500',  # Bright Orange
    'Normal': '#800080'  # Bright Purple
}

# Visualize one WSI graph with WSI background
def visualize_graph_with_wsi(name_wsi, graph, wsi_image_path=None, alpha=0.5):
    """Visualize the KNN graph on top of the WSI image with transparency."""
    if wsi_image_path:
        slide = openslide.OpenSlide(wsi_image_path)
        slide_dim = slide.dimensions
        plt.figure(figsize=(50, 50))
        wsi_image = slide.read_region((0, 0), 0, slide_dim).convert('RGBA')  # Convert to RGBA for transparency
        wsi_image.putalpha(int(alpha * 255))  # Apply transparency

        plt.imshow(wsi_image)
        pos = nx.get_node_attributes(graph, 'pos')
        node_colors = [class_colors[graph.nodes[n]['label']] for n in graph.nodes]
        nx.draw(
            graph,
            pos,
            node_size=100,
            node_color=node_colors,
            edge_color='cyan',  # Cyan for edges
            alpha=0.8,
            width=1.0,
            with_labels=False,
            ax=plt.gca()
        )
        plt.title(f"Graph for WSI: {name_wsi}")
        figure_save_path = f"/home/akebli/test5/try/graph_{name_wsi}_with_wsi.png"
        plt.savefig(figure_save_path, bbox_inches='tight')
        plt.show()
        print(f"Graph for WSI {name_wsi} saved to {figure_save_path}")
    else:
        print("No WSI image path provided.")



# Visualize only the graph with the same dimensions as the WSI
def visualize_graph_only(name_wsi, graph, wsi_image_path=None):
    """Visualize only the KNN graph without the WSI image, keeping the same dimensions."""
    if wsi_image_path:
        slide = openslide.OpenSlide(wsi_image_path)
        slide_dim = slide.dimensions
        plt.figure(figsize=(50, 50))
        
        # Get the graph positions and set the axis limits to match the WSI dimensions
        pos = nx.get_node_attributes(graph, 'pos')
        plt.xlim(0, slide_dim[0])
        plt.ylim(slide_dim[1], 0)  # Reverse y-axis to match image coordinates
        
        # Ensure the nodes and edges are drawn correctly
        node_colors = [class_colors[graph.nodes[n]['label']] for n in graph.nodes]
        nx.draw(
            graph,
            pos,
            node_size=100,
            node_color=node_colors,
            edge_color='cyan',  # Cyan for edges
            alpha=0.8,
            width=1.0,
            with_labels=False,
            ax=plt.gca()
        )
        plt.title(f"Graph for WSI: {name_wsi} (Only Graph)")
        figure_save_path = f"/home/akebli/test5/try/graph_{name_wsi}_only_graph.png"
        plt.savefig(figure_save_path, bbox_inches='tight')
        plt.show()
        print(f"Graph for WSI {name_wsi} (only graph) saved to {figure_save_path}")
    else:
        print("No WSI image path provided.")


# Select a WSI ID to visualize
wsi_name_to_visualize = 'Subset1_Train_49'
print(f"Visualizing graph for WSI: {wsi_name_to_visualize}")

# Path to the WSI image file
wsi_image_path = f"/mnt/dmif-nas/MITEL/challenges/AGGC22/ProMaL/slides/{wsi_name_to_visualize}.tiff"
# Visualize the graph with WSI background
visualize_graph_with_wsi(wsi_name_to_visualize, graphs[wsi_name_to_visualize], wsi_image_path=wsi_image_path,alpha=0.2)
# Visualize the graph alone with same dimensions
visualize_graph_only(wsi_name_to_visualize, graphs[wsi_name_to_visualize], wsi_image_path=wsi_image_path)

print("Done")
