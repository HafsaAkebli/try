import torch
import numpy as np
from torch_geometric.data import Data
import os

# Define the device to use (GPU if available, otherwise CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("CUDA is available:", torch.cuda.is_available())

# Import the necessary functions (assuming the correct path)
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

# Normalize the features
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
features = scaler.fit_transform(features)

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

    # Clip edge weights to ensure non-negative values
    edge_weights = torch.clamp(edge_weights, min=0)

    # Check for NaNs and Infs in node features, edge weights, and node labels
    if torch.isnan(node_features).any() or torch.isinf(node_features).any():
        raise ValueError("Node features contain NaNs or Infs")
    if torch.isnan(edge_weights).any() or torch.isinf(edge_weights).any():
        raise ValueError("Edge weights contain NaNs or Infs")
    if torch.isnan(node_labels).any() or torch.isinf(node_labels).any():
        raise ValueError("Node labels contain NaNs or Infs")

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

# Function to check edge weights, node features, and node labels
def check_graph_data(data_list, output_file):
    overall_edge_weight_min = np.inf
    overall_edge_weight_max = -np.inf
    overall_edge_weight_sum = 0
    overall_edge_weight_count = 0

    overall_feature_min = np.inf
    overall_feature_max = -np.inf
    overall_feature_sum = 0
    overall_feature_count = 0

    class_counts = {cls: 0 for cls in class_colors.keys()}  # Initialize counts for each class

    with open(output_file, 'w') as f:
        for idx, data in enumerate(data_list):
            edge_weights = data.edge_attr.cpu().numpy()
            node_features = data.x.cpu().numpy()
            node_labels = data.y.cpu().numpy()

            # Check for NaNs and Infs in edge weights and node features
            if np.isnan(edge_weights).any() or np.isinf(edge_weights).any():
                f.write(f"Graph {idx}: Edge weights contain NaNs or Infs.\n")
            if np.isnan(node_features).any() or np.isinf(node_features).any():
                f.write(f"Graph {idx}: Node features contain NaNs or Infs.\n")

            # Update edge weight statistics
            overall_edge_weight_min = min(overall_edge_weight_min, edge_weights.min())
            overall_edge_weight_max = max(overall_edge_weight_max, edge_weights.max())
            overall_edge_weight_sum += edge_weights.sum()
            overall_edge_weight_count += edge_weights.size

            # Update node feature statistics
            overall_feature_min = min(overall_feature_min, node_features.min())
            overall_feature_max = max(overall_feature_max, node_features.max())
            overall_feature_sum += node_features.sum()
            overall_feature_count += node_features.size

            # Update class counts
            for label in node_labels:
                class_name = [k for k, v in class_to_index.items() if v == label][0]
                class_counts[class_name] += 1

            # Write data consistency for each graph to file
            f.write(f"Graph {idx}:\n")
            f.write(f"Number of nodes: {data.num_nodes}\n")
            f.write(f"Number of edges: {data.num_edges}\n")
            f.write(f"Number of features: {data.x.size(1)}\n")
            f.write(f"Edge weights: min={edge_weights.min()}, max={edge_weights.max()}\n")
            f.write(f"Node features: min={node_features.min()}, max={node_features.max()}\n")
            f.write(f"Node labels: min={node_labels.min()}, max={node_labels.max()}\n\n")

        # Write overall statistics to file
        overall_edge_weight_mean = overall_edge_weight_sum / overall_edge_weight_count
        overall_feature_mean = overall_feature_sum / overall_feature_count
        f.write(f"Overall edge weights: min={overall_edge_weight_min}, max={overall_edge_weight_max}, mean={overall_edge_weight_mean}\n")
        f.write(f"Overall node features: min={overall_feature_min}, max={overall_feature_max}, mean={overall_feature_mean}\n")

        # Write class distribution to file
        f.write("\nClass Distribution:\n")
        for cls, count in class_counts.items():
            f.write(f"{cls}: {count} nodes\n")

# Specify the output file for inspection results
output_file = "/home/akebli/test5/graph_data_inspection.txt"

# Call the function to check edge weights, node features, and node labels, and write to file
check_graph_data(data_list, output_file)

print("Data inspection completed. Results written to", output_file)
