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
sys.stdout = open('validation_results_150_32_0.001.txt', 'w')

from build_graphs_cosine import load_features, organize_patches_by_wsi, build_graph_for_wsi

# The files of the patches features 
subset_names = ['Subset1', 'Subset3']
output_features_files = [f"/home/akebli/test5/features_{subset_name}_train_prostate_medium.npz" for subset_name in subset_names]
output_features_files_valid = [f"/home/akebli/test5/features_{subset_name}_valid_prostate_medium.npz" for subset_name in subset_names]

# Load features, labels, and patch paths for training
print("Loading training features...")
train_features, train_labels, train_patch_paths = load_features(output_features_files)
train_features = np.array(train_features)
train_labels = np.array(train_labels)
train_patch_paths = np.array(train_patch_paths)

input_dim = train_features.shape[1]
print("The number of input dimensions is", input_dim)

# Load features, labels, and patch paths for validation
print("Loading validation features...")
valid_features, valid_labels, valid_patch_paths = load_features(output_features_files_valid)
valid_features = np.array(valid_features)
valid_labels = np.array(valid_labels)
valid_patch_paths = np.array(valid_patch_paths)

# Normalize the node features using the same scaler for both training and validation
scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
valid_features = scaler.transform(valid_features)

# Create a mapping from patch path to the patch label or class
train_patch_to_label = dict(zip(train_patch_paths, train_labels))
valid_patch_to_label = dict(zip(valid_patch_paths, valid_labels))

# Organize patches by WSI for training
print("Organizing training patches by WSI...")
train_wsi_patches = organize_patches_by_wsi(train_patch_paths, train_patch_to_label)

# Organize patches by WSI for validation
print("Organizing validation patches by WSI...")
valid_wsi_patches = organize_patches_by_wsi(valid_patch_paths, valid_patch_to_label)

# Create patch to feature mapping for training
train_patch_to_feature = {train_patch_paths[i]: train_features[i] for i in range(len(train_patch_paths))}
print("Training patch to feature mapping created.")

# Create patch to feature mapping for validation
valid_patch_to_feature = {valid_patch_paths[i]: valid_features[i] for i in range(len(valid_patch_paths))}
print("Validation patch to feature mapping created.")

# Build graphs using cosine similarity and patches as nodes for training
print("Building training graphs...")
train_graphs = build_graph_for_wsi(train_wsi_patches, train_patch_to_feature)
print("Training graphs built.")

# Build graphs using cosine similarity and patches as nodes for validation
print("Building validation graphs...")
valid_graphs = build_graph_for_wsi(valid_wsi_patches, valid_patch_to_feature)
print("Validation graphs built.")

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

# Convert all graphs to PyTorch Geometric Data objects for training
def convert_graphs_to_data_list(graphs, class_labels):
    data_list = []
    for graph in graphs.values():
        data = convert_graph_to_data(graph, class_labels)
        data_list.append(data)
    return data_list

print("Converting training graphs to data list...")
train_data_list = convert_graphs_to_data_list(train_graphs, class_to_index)
print("Training graphs converted to data list.")

print("Converting validation graphs to data list...")
valid_data_list = convert_graphs_to_data_list(valid_graphs, class_to_index)
print("Validation graphs converted to data list.")

# Create DataLoaders for training and validation
train_loader = DataLoader(train_data_list, batch_size=32, shuffle=True)
val_loader = DataLoader(valid_data_list, batch_size=32, shuffle=False)
print("DataLoaders created.")

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
criterion = nn.CrossEntropyLoss() #cross entropy loss function
print("Criterion initialized.")
optimizer = optim.Adam(model.parameters(), lr=0.001) #adam optimizer
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
def train(model, train_loader, val_loader, criterion, optimizer, epochs=150):
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
        
        # Validate after each epoch
        accuracy, precision, recall, f1 = validate(model, val_loader)
        
        # Save the model if validation F1 score improves
        if f1 > best_f1:
            best_f1 = f1
            model_save_path = "/home/akebli/test5/try/gcn_model_test_150_32_0.001.pth"
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved to {model_save_path}")

print("Starting training...")
train(model, train_loader, val_loader, criterion, optimizer, epochs=150)
print("Training completed.")

# Close the text file to save the prints
sys.stdout.close()
