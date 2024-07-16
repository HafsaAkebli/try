import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader
import numpy as np
from torch_geometric.data import Data
import networkx as nx

# Define the device to use (GPU if available, otherwise CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("CUDA is available:", torch.cuda.is_available())

from build_graphs_cosine import load_features, organize_patches_by_wsi, build_graph_for_wsi

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
    'Normal': '#800080'  # Bright Purple
}

class_to_index = {cls: i for i, cls in enumerate(class_colors.keys())}
print("Class to index mapping created.")

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

import matplotlib.pyplot as plt
import seaborn as sns

#visualize node features
def visualize_node_features(data_list):
    for idx, data in enumerate(data_list):
        node_features = data.x.cpu().numpy()
        
        plt.figure(figsize=(12, 6))
        plt.title(f'Node Features Distribution for Graph {idx}')
        sns.histplot(node_features.flatten(), kde=True)
        plt.xlabel('Feature Value')
        plt.ylabel('Frequency')
        plt.show()

# Call the function to visualize node features
visualize_node_features(data_list)

#inspect the distribution of edges weights

def inspect_edge_weights(data_list):
    for idx, data in enumerate(data_list):
        edge_weights = data.edge_attr.cpu().numpy()
        
        if np.isnan(edge_weights).any() or np.isinf(edge_weights).any():
            print(f"Graph {idx}: Edge weights contain NaNs or Infs.")
        else:
            print(f"Graph {idx}: Edge weights do not contain NaNs or Infs.")
        
        print(f"Graph {idx}: Edge weights range: min={edge_weights.min()}, max={edge_weights.max()}")
        
        # Visualize the edge weights
        plt.figure(figsize=(12, 6))
        plt.title(f'Edge Weights Distribution for Graph {idx}')
        sns.histplot(edge_weights, kde=True)
        plt.xlabel('Edge Weight')
        plt.ylabel('Frequency')
        plt.show()

# Call the function to inspect edge weights
inspect_edge_weights(data_list)

#check data consistency
def check_graph_data_consistency(data_list):
    for idx, data in enumerate(data_list):
        print(f"Graph {idx}:")
        print(f"Number of nodes: {data.num_nodes}")
        print(f"Number of edges: {data.num_edges}")
        print(f"Number of features: {data.x.size(1)}")
        print(f"Edge weights: min={data.edge_attr.min()}, max={data.edge_attr.max()}")
        print(f"Node labels: min={data.y.min()}, max={data.y.max()}")
        print(f"Batch size: {data.batch.size(0)}")  # Should be 1 due to batch_size=1
        print()

# Call the function to check data consistency
check_graph_data_consistency(data_list)