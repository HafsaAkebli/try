
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader
import numpy as np
from torch_geometric.data import Data
import networkx as nx
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import os
from collections import defaultdict
from PIL import Image
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

# Disable the decompression bomb check for large images
Image.MAX_IMAGE_PIXELS = None

# Define your output features file (replace with the correct subset names)
subset_names = ['Subset1', 'Subset3']
output_features_files = [f"/home/akebli/test5/features_{subset_name}_valid_prostate_medium.npz" for subset_name in subset_names]

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
def build_graph_for_wsi(wsi_patches, patch_to_feature, k=5):
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

        # Check for edges without weights
        for u, v, data in G.edges(data=True):
            if 'weight' not in data:
                print(f"Edge ({u}, {v}) does not have a 'weight' attribute.")

        graphs[wsi_id] = G
    return graphs

# Build graphs for each WSI
graphs = build_graph_for_wsi(wsi_patches,patch_to_feature)
print(f"Graphs have been built for {len(graphs)} WSIs.")



# Define the device to use (GPU if available, otherwise CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("CUDA is available:", torch.cuda.is_available())

from build_graphs_cosine import load_features, organize_patches_by_wsi, build_graph_for_wsi

# Define your output features file (replace with the correct subset names)
subset_names = ['Subset1', 'Subset3']
output_features_files = [f"/home/akebli/test5/features_{subset_name}_train_prostate_medium.npz" for subset_name in subset_names]

# Load features, labels, and patch paths
print("Loading features extracted by histoencoder...")
features, labels, patch_paths = load_features(output_features_files)
features = np.array(features)
labels = np.array(labels)
patch_paths = np.array(patch_paths)
input_dim = features.shape[1]
print("The number of input dimensions is", input_dim)

# Normalize the node features
scaler = StandardScaler()
features = scaler.fit_transform(features) #standardize features with zero mean and unit variance

# Group Patches by WSI
print("Organizing patches by WSI...")
wsi_patches = organize_patches_by_wsi(patch_paths)

# Create patch to feature mapping
patch_to_feature = {patch_paths[i]: features[i] for i in range(len(patch_paths))}
print("Patch to feature mapping created.")

# Build graphs using cosine similarity and patches as nodes
print("Building graphs...")
graphs = build_graph_for_wsi(wsi_patches, patch_to_feature)
print("Graphs built.")

# Define class to index mapping
class_colors = {
    'G3': '#FF0000',  # Bright Red
    'G4': '#00FF00',  # Bright Green
    'G5': '#0000FF',  # Bright Blue
    'Stroma': '#FFA500',  # Bright Orange
    'Normal': '#800080'  # Bright Purple
}

class_to_index = {cls: i for i, cls in enumerate(class_colors.keys())}
print("Class to index mapping created.")

# Preprocess edge weights
def preprocess_edge_weights(weights):
    scaler = MinMaxScaler(feature_range=(0, 1))  # Rescale weights to [0, 1]
    weights = np.array(weights).reshape(-1, 1)  # Convert to 2D array
    weights = scaler.fit_transform(weights).flatten()  # Rescale and flatten
    return weights

# Convert NetworkX graph to PyTorch Geometric Data object
def convert_graph_to_data(graph, class_labels):
    node_features = [graph.nodes[node]['feature'] for node in graph.nodes]
    node_labels = [class_labels[graph.nodes[node]['label']] for node in graph.nodes]
    edge_indices = []
    edge_weights = []

    for edge in graph.edges:
        src, dst = edge
        edge_indices.append((list(graph.nodes).index(src), list(graph.nodes).index(dst)))
        edge_weights.append(graph.edges[edge]['weight'])

    # Preprocess edge weights
    edge_weights = preprocess_edge_weights(edge_weights)

    # Convert lists to tensors
    node_features = torch.tensor(np.array(node_features), dtype=torch.float).to(device)
    edge_indices = torch.tensor(np.array(edge_indices).T, dtype=torch.long).to(device)
    edge_weights = torch.tensor(np.array(edge_weights), dtype=torch.float).to(device)
    node_labels = torch.tensor(np.array(node_labels), dtype=torch.long).to(device)

    data = Data(x=node_features, edge_index=edge_indices, edge_attr=edge_weights, y=node_labels)
    return data

# Convert all graphs to PyTorch Geometric Data objects
def convert_graphs_to_data_list(graphs, class_labels):
    data_list = []
    for graph in graphs.values():
        data = convert_graph_to_data(graph, class_labels)
        data_list.append(data)
    return data_list

print("Converting graphs to data list...")
data_list = convert_graphs_to_data_list(graphs, class_to_index)
print("Graphs converted to data list.")
