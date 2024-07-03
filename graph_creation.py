import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors 

# Path to the .npz file
output_features_file = "/home/akebli/test5/features_prostate_medium.npz"

# Load the .npz file
data = np.load(output_features_file)

# Access the features and labels arrays
features = data['features']
labels = data['labels']

# Number of neighbors
k = 5

# Fit the NearestNeighbors model using Euclidean distance
nn_model = NearestNeighbors(n_neighbors=k + 1, metric='euclidean')
nn_model.fit(features)

# Find the k-nearest neighbors for each point
distances, indices = nn_model.kneighbors(features)

# Create a graph
G = nx.Graph()

# Add nodes with labels as attributes
for i, label in enumerate(labels):
    G.add_node(i, label=label)

# Add edges based on k-nearest neighbors
for i, neighbors in enumerate(indices):
    for neighbor in neighbors[1:]:  # Skip the first neighbor since it's the node itself
        G.add_edge(i, neighbor)

# Visualize the graph
plt.figure(figsize=(12, 12))
pos = nx.spring_layout(G)  # Layout for visualization
nx.draw_networkx_nodes(G, pos, node_size=50, node_color=labels, cmap=plt.cm.viridis)
nx.draw_networkx_edges(G, pos, alpha=0.5)
plt.title("KNN Graph of Patch Features")
plt.show()


# Save the figure
figure_save_path = "/home/akebli/test5/try/graph1.png"
plt.savefig(figure_save_path)

# Show the figure
plt.show()
