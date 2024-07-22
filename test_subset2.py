import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader
import numpy as np
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from collections import defaultdict
import sys

# Define the device to use (GPU if available, otherwise CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("CUDA is available:", torch.cuda.is_available())

# Redirect prints to a text file
sys.stdout = open('testing_results_1.txt', 'w')

model_path = "/home/akebli/test5/model_kfold_150_0.001_32.pth"

from build_graphs_cosine import load_features, organize_patches_by_wsi, build_graph_for_wsi

# The files of the patches features 
subset_names = ['Subset2']
output_features_files_subset2 = ["/home/akebli/test5/features_Subset2_1.npz"]

# Load features, labels, and patch paths for testing
print("Loading test features...")
test_features, test_labels, test_patch_paths = load_features(output_features_files_subset2)
test_features = np.array(test_features)
test_labels = np.array(test_labels)
test_patch_paths = np.array(test_patch_paths)

# Normalize the node features using the same scaler for both training and testing
scaler = StandardScaler()
test_features = scaler.fit_transform(test_features)
test_patch_to_label = dict(zip(test_patch_paths, test_labels))

# Create patch to feature mapping for testing
test_patch_to_feature = {test_patch_paths[i]: test_features[i] for i in range(len(test_patch_paths))}
print("Test patch to feature mapping created.")

# Organize patches by WSI for testing
print("Organizing test patches by WSI...")
test_wsi_patches = organize_patches_by_wsi(test_patch_paths, test_patch_to_label)

# Build graphs using cosine similarity and patches as nodes for testing
print("Building test graphs...")
test_graphs = build_graph_for_wsi(test_wsi_patches, test_patch_to_feature)
print("Test graphs built.")

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

# Convert all graphs to PyTorch Geometric Data objects for testing
def convert_graphs_to_data_list(graphs, class_labels):
    data_list = []
    for graph in graphs.values():
        data = convert_graph_to_data(graph, class_labels)
        data_list.append(data)
    return data_list

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

print("Converting testing graphs to data list...")
test_data_list = convert_graphs_to_data_list(test_graphs, class_to_index)
print("Testing graphs converted to data list.")

# Create DataLoader for testing
test_loader = DataLoader(test_data_list, batch_size=32, shuffle=False)
print("Test DataLoader created.")

# Define the GCN model
class GCNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_weight=edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_weight=edge_attr))
        x = self.fc(x)  # Node-level predictions
        return x

# Load the trained model
hidden_dim = 64  # Hidden layer dimension
output_dim = len(class_colors)  # Number of classes
input_dim = test_data_list[0].x.shape[1]  # Number of input features
print("input dimension of GCN",input_dim)
model = GCNModel(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
model.load_state_dict(torch.load(model_path))
model.to(device)
model.eval()
print("Model loaded successfully.")

# Predict and calculate metrics
all_preds = []
all_labels = []

with torch.no_grad():
    for data in test_loader:
        data = data.to(device)
        output = model(data)
        preds = output.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(data.y.cpu().numpy())

# Calculate metrics
accuracy = accuracy_score(all_labels, all_preds)
precision, recall, f1_score, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1_score:.4f}")

# Redirect back to console
sys.stdout.close()
sys.stdout = sys.__stdout__
