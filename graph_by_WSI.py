import numpy as np
import os
from PIL import Image
import networkx as nx
import faiss
import matplotlib.pyplot as plt
from PIL import Image


# Load the .npz file containing features, labels, and patch paths
output_features_file = "/home/akebli/test5/features_prostate_medium.npz"
data = np.load(output_features_file)
features = data['features']
labels = data['labels']
patch_paths = data['patch_paths']

# Create a mapping from patch path to feature vector
patch_to_feature = dict(zip(patch_paths, features))

# Function to extract WSI name from patch patches names
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
        patch_features = np.array([patch_to_feature[patch] for patch in patches])
        index = faiss.IndexFlatL2(patch_features.shape[1])
        index.add(patch_features)
        distances, indices = index.search(patch_features, k + 1)
        
        G = nx.Graph()
        for i, patch_path in enumerate(patches):
            G.add_node(patch_path, feature=patch_to_feature[patch_path])
            for j in range(1, k + 1):
                neighbor_idx = indices[i, j]
                neighbor_patch = patches[neighbor_idx]
                distance = distances[i, j]
                G.add_edge(patch_path, neighbor_patch, weight=distance)
        
        graphs[wsi_id] = G
    return graphs

# Build graphs for each WSI
graphs = build_graph_for_wsi(wsi_patches)

# Visualize one WSI graph
def visualize_graph(name_wsi, graph, wsi_image_path=None):
    pos = nx.spring_layout(graph, weight='weight')
    plt.figure(figsize=(12, 12))
    nx.draw(graph, pos, node_size=50, with_labels=False)
    if wsi_image_path:
        wsi_image = Image.open(wsi_image_path)
        plt.imshow(wsi_image, alpha=0.5)
    plt.title(f"Graph for WSI: {name_wsi}")
    plt.show()
    # Save the figure
    figure_save_path = (f"/home/akebli/test5/try/graph{name_wsi}.png")
    plt.savefig(figure_save_path)

    # Show the figure
    plt.show()

# Select a WSI ID to visualize

visualize_graph('Subset1_Train_4', graphs['Subset1_Train_4'])
