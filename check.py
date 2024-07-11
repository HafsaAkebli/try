import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader
import numpy as np
from torch_geometric.data import Data
import networkx as nx
from build_graphs_cosine import load_features, organize_patches_by_wsi, build_graph_for_wsi

# Define the device to use (GPU if available, otherwise CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("CUDA is available:", torch.cuda.is_available())

# Define your output features file (replace with the correct subset names)
subset_names = ['Subset1', 'Subset3']
output_features_files = [f"/home/akebli/test5/features_{subset_name}_train_prostate_medium.npz" for subset_name in subset_names]

# Load features, labels, and patch paths
print("Loading features...")
features, labels, patch_paths = load_features(output_features_files)
features = np.array(features)
labels = np.array(labels)
patch_paths = np.array(patch_paths)
input_dim = features.shape[1]
print("The number of input dimensions is", input_dim)

# Organize patches by WSI
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
    'Normal': '#800080'}  # Bright Purple 

class_to_index = {cls: i for i, cls in enumerate(class_colors.keys())}
print("Class to index mapping created.")

# Convert NetworkX graph to PyTorch Geometric Data object
def convert_graph_to_data(graph, class_labels):
    """Convert a NetworkX graph to a PyTorch Geometric Data object."""
    node_features = []
    edge_indices = []
    edge_weights = []
    node_labels = []

    for i, node in enumerate(graph.nodes):
        node_features.append(graph.nodes[node]['feature'])
        node_labels.append(class_labels[graph.nodes[node]['label']])

    for edge in graph.edges:
        src, dst = edge
        edge_indices.append((list(graph.nodes).index(src), list(graph.nodes).index(dst)))
        if 'weight' in graph.edges[edge]:
            edge_weights.append(graph.edges[edge]['weight'])
        else:
            print(f"Edge {edge} does not have a 'weight' attribute.")

    # Convert lists to numpy arrays before creating tensors
    node_features = np.array(node_features)
    edge_indices = np.array(edge_indices)
    edge_weights = np.array(edge_weights)

    x = torch.tensor(node_features, dtype=torch.float).to(device)
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous().to(device)
    edge_attr = torch.tensor(edge_weights, dtype=torch.float).view(-1, 1).to(device)
    y = torch.tensor(node_labels, dtype=torch.long).to(device)

    # Adding batch information
    batch = torch.zeros(x.size(0), dtype=torch.long).to(device)  # All nodes in one graph

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, batch=batch)
    return data

# Convert all graphs to PyTorch Geometric Data objects
def convert_graphs_to_data_list(graphs, class_labels):
    """Convert all NetworkX graphs to a list of PyTorch Geometric Data objects."""
    data_list = []
    for graph in graphs.values():
        data = convert_graph_to_data(graph, class_labels)
        data_list.append(data)
    return data_list

print("Converting graphs to data list...")
data_list = convert_graphs_to_data_list(graphs, class_to_index)
print("Graphs converted to data list.")

# DataLoader
loader = DataLoader(data_list, batch_size=32, shuffle=True)
print("DataLoader created.")