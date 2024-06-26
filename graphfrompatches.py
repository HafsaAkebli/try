import torch



# define the device (GPU, else CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())


#read the patches from the server that I created 
patches_sub="/home/akebli/test5/patches/"
import networkx as nx
import numpy as np
from sklearn.neighbors import kneighbors_graph
from torch_geometric.data import Data

# Assume 'patches' is a list of feature vectors extracted from patches
features = np.random.randn(len(patches), 64)  # Example feature vectors

# Construct a k-NN graph based on feature similarity
k = 5  # Number of nearest neighbors
knn_graph = kneighbors_graph(features, k, mode='connectivity', include_self=False)

# Convert k-NN graph to NetworkX format
graph = nx.from_scipy_sparse_matrix(knn_graph)

# Define a basic GNN model using PyTorch Geometric
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Dummy data for demonstration
x = torch.randn(len(features), features.shape[1])  # Feature vectors as input
edge_index = torch.tensor(list(graph.edges)).t().contiguous()

# Initialize and train the GNN model
model = GCN(input_dim=features.shape[1], hidden_dim=64, output_dim=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def train():
    model.train()
    optimizer.zero_grad()
    log_logits = model(x, edge_index)
    loss = F.nll_loss(log_logits[train_mask], labels[train_mask])
    loss.backward()
    optimizer.step()

# Example training loop
for epoch in range(100):
    train()

