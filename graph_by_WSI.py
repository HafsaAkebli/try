import os
from collections import defaultdict
import networkx as nx
import faiss
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

#extract the name of the WSI
def extract_wsi_id(filename):
    parts = filename.split('-')
    return parts[0]

def organize_patches_by_wsi(patch_folder):
    wsi_patches = defaultdict(list)
    for cls in ["G3", "G4", "G5", "Stroma", "Normal"]:
        cls_folder = os.path.join(patch_folder, cls)
        for patch_file in os.listdir(cls_folder):
            if patch_file.endswith(".jpg"):
                wsi_id = extract_wsi_id(patch_file)
                patch_path = os.path.join(cls_folder, patch_file)
                wsi_patches[wsi_id].append(patch_path)
    return wsi_patches

patch_folder = "/home/akebli/test5/patches/"
wsi_patches = organize_patches_by_wsi(patch_folder)

# Load the .npz file containing features and labels
output_features_file = "/home/akebli/test5/features_prostate_medium.npz"
data = np.load(output_features_file)
features = data['features']
labels = data['labels']

# Create a mapping from patch path to feature vector
patch_to_feature = dict(zip(data['patch_paths'], features))

#Build graph for each WSI using it's patches and their feature vectors, with KNN
def build_graph_for_wsi(wsi_patches, k=5):
    patch_features = []
    patch_paths = []
    for patch_path in wsi_patches:
        patch_features.append(patch_to_feature[patch_path])
        patch_paths.append(patch_path)
    
    patch_features = np.array(patch_features).astype('float32')
    
    res = faiss.StandardGpuResources()
    index_flat = faiss.IndexFlatL2(patch_features.shape[1])
    gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
    gpu_index_flat.add(patch_features)
    
    distances, indices = gpu_index_flat.search(patch_features, k + 1)
    
    G = nx.Graph()
    for i, patch_path in enumerate(patch_paths):
        G.add_node(patch_path, label=labels[i])
    
    for i, neighbors in enumerate(indices):
        for neighbor in neighbors[1:]:
            G.add_edge(patch_paths[i], patch_paths[neighbor])
    
    return G

# Build graphs for all WSIs
wsi_graphs = {}
for wsi_id, patches in wsi_patches.items():
    wsi_graphs[wsi_id] = build_graph_for_wsi(patches)


#visualize a graph for one WSI
def visualize_graph(graph, sample_size=500):
    sampled_nodes = random.sample(list(graph.nodes()), k=min(sample_size, len(graph.nodes())))
    sampled_graph = graph.subgraph(sampled_nodes)
    
    features = []
    labels = []
    for node in sampled_graph.nodes:
        features.append(patch_to_feature[node])
        labels.append(sampled_graph.nodes[node]['label'])
    
    features = np.array(features)
    
    pca = PCA(n_components=50)
    pca_result = pca.fit_transform(features)
    
    tsne = TSNE(n_components=2, random_state=0)
    tsne_result = tsne.fit_transform(pca_result)
    
    df = pd.DataFrame(tsne_result, columns=['Component 1', 'Component 2'])
    df['label'] = labels
    
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(df['Component 1'], df['Component 2'], c=df['label'], cmap='viridis', alpha=0.5)
    plt.colorbar(scatter, label='Class')
    plt.title('t-SNE Visualization of Extracted Features')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.show()

# Visualize a specific WSI's graph
visualize_graph(wsi_graphs['Subset1_Train_4'])  # Replace with an actual WSI ID
