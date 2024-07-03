import numpy as np
import os
import random
from collections import defaultdict
from PIL import Image
import networkx as nx
import matplotlib.pyplot as plt
import torch
from sklearn.neighbors import NearestNeighbors  # Remove if using FAISS

# Define your output features file
output_features_file = "/home/akebli/test5/features_prostate_medium.npz"

# Load the .npz file containing features, labels, and patch paths
data = np.load(output_features_file)
features = data['features']
labels = data['labels']
patch_paths = data['patch_paths']

# Create a mapping from patch path to feature vector
patch_to_feature = dict(zip(patch_paths, features))

# Function to extract WSI name from patch paths
def extract_name_wsi(filename):
    parts = filename.split('-')
    return parts[0]

# Organize patches by WSI
def organize_patches_by_wsi(patch_paths):
    wsi_patches = defaultdict(list)
    for patch_path in patch_paths:
        wsi_id = extract_name_wsi(os.path.basename(patch_path))
        wsi_patches[wsi_id].append(patch_path)
    return wsi_patches

# Organize patches by WSI
wsi_patches = organize_patches_by_wsi(patch_paths)

# Build graph for each WSI
def build_graph_for_wsi(wsi_patches, k=5):
    graphs = {}
    for wsi_id, patches in wsi_patches.items():
        # Extract features for the patches
        patch_features = np.array([patch_to_feature[patch] for patch in patches])

        # Initialize NearestNeighbors and fit the model
        nn = NearestNeighbors(n_neighbors=k + 1, metric='euclidean')
        nn.fit(patch_features)
        distances, indices = nn.kneighbors(patch_features)

        G = nx.Graph()
        for i, patch_path in enumerate(patches):
            G.add_node(patch_path, feature=patch_to_feature[patch_path])
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
    pos = nx.spring_layout(graph, weight='weight', seed=42)
    plt.figure(figsize=(12, 12))

    # Draw the graph
    nx.draw(graph, pos, node_size=50, with_labels=False)

    if wsi_image_path:
        # Open and show the whole slide image
        wsi_image = Image.open(wsi_image_path)
        plt.imshow(wsi_image, alpha=0.5)

    plt.title(f"Graph for WSI: {name_wsi}")

    # Save the figure
    figure_save_path = f"/home/akebli/test5/try/graph_{name_wsi}.png"
    plt.savefig(figure_save_path)
    plt.show()
    print(f"Graph for WSI {name_wsi} saved to {figure_save_path}")

# Select a WSI ID to visualize
wsi_id_to_visualize = 'Subset1_Train_49'
print(f"Visualizing graph for WSI: {wsi_id_to_visualize}")

# Path to the WSI image file
wsi_image_path = f"/mnt/dmif-nas/MITEL/challenges/AGGC22/ProMaL/slides/{wsi_id_to_visualize}.tiff"
visualize_graph(wsi_id_to_visualize, graphs[wsi_id_to_visualize], wsi_image_path=wsi_image_path)

print("done")