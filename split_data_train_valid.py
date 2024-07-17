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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from collections import defaultdict
import random

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

# Normalize the node features
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


# Organize data by WSI
wsi_data = defaultdict(list)
for graph, wsi in zip(data_list, wsi_patches.keys()):
    wsi_data[wsi].append(graph)

# Split WSI into training and validation sets
wsi_keys = list(wsi_patches.keys())
random.shuffle(wsi_keys)
split_idx = int(len(wsi_keys) * 0.8)
train_wsis = wsi_keys[:split_idx]
val_wsis = wsi_keys[split_idx:]

train_data = [graph for wsi in train_wsis for graph in wsi_data[wsi]]
val_data = [graph for wsi in val_wsis for graph in wsi_data[wsi]]

# Split data into training and validation sets
train_data, val_data = train_test_split(data_list, test_size=0.2, random_state=42)
train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
val_loader = DataLoader(val_data, batch_size=1, shuffle=False)
print("DataLoader created.")

# Define the GCN model
class GCNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(p=0.5)  # Add dropout layer with 50% dropout rate

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_weight=edge_attr))
        x = self.dropout(x)  # Apply dropout
        x = F.relu(self.conv2(x, edge_index, edge_weight=edge_attr))
        x = self.fc(x)  # Node-level predictions
        return x

# Define the model, loss function, and optimizer
hidden_dim = 64  # Hidden layer dimension
output_dim = len(class_colors)  # Number of classes

model = GCNModel(input_dim, hidden_dim, output_dim).to(device)
print("Model initialized.")
criterion = nn.CrossEntropyLoss()
print("Criterion initialized.")
optimizer = optim.Adam(model.parameters(), lr=0.001)
print("Model, criterion, and optimizer initialized.")

# Validate the model on the validation set
def validate(model, val_loader):
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)
            out = model(data)
            preds = out.argmax(dim=1)
            all_labels.extend(data.y.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
    print(f'Validation Accuracy: {accuracy:.4f}')
    print(f'Validation Precision: {precision:.4f}')
    print(f'Validation Recall: {recall:.4f}')
    print(f'Validation F1 Score: {f1:.4f}')
    return accuracy, precision, recall, f1

# Training loop
def train(model, train_loader, val_loader, criterion, optimizer, epochs=10):
    best_f1 = 0  # Best F1 score for model saving
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for data in train_loader:
            optimizer.zero_grad()
            data = data.to(device)  # Move data to GPU
            out = model(data)
            loss = criterion(out, data.y)
            if torch.isnan(loss):
                print("Loss is NaN!")
                return
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}')
        accuracy, precision, recall, f1 = validate(model, val_loader)
        # Save the model if validation F1 score improves
        if f1 > best_f1:
            best_f1 = f1
            model_save_path = "/home/akebli/test5/try/gcn_model_3_Patches_KNN_Cosine_best.pth"
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved to {model_save_path}")

print("Starting training...")
train(model, train_loader, val_loader, criterion, optimizer, epochs=100)
print("Training completed.")

# Save the final trained model
model_save_path = "/home/akebli/test5/try/gcn_model_4_Patches_KNN_Cosine_final.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Final model saved to {model_save_path}")