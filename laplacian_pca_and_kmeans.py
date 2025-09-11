import numpy as np
import os
import numpy.linalg as npl
import scipy.io as io
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.cluster import KMeans
import librosa
import pandas as pd



'''

j_i is a feature vector for the ith audio file j_i \in R^{1 x 129} (129 is step choice) (accidentally added zeroth index on for loop)

use cosine kernel to compute ijth value of weight matrix 

x_i^T x_j / ||x_i|| ||x_kj|| = cos(theta)

map negative values to zero


FINAL_NOTE ----- THIS FILE IS not complete!! I need to map the clusters to their physical features via the pca weights

'''


def cos_kernel(feature_matrix): # input feature vectors and perfrom (ab^T/ ||a|| ||b||)
    num = feature_matrix.T @ feature_matrix # inner product
    denom = npl.norm(feature_matrix)
    denom = denom**2
    return num/denom 


def cos_affinity(feature_matrix): 
    W = cos_kernel(feature_matrix) # output is unit vector 
    W[W < 0] = 0 # map all negative valyes to zero
    np.fill_diagonal(W, 0)
    W = (W + W.T) / 2
    return W


def laplacian_matr(W):  #  L = D - W
    degrees = np.sum(W, axis=1) # sum over cols (equal to sum over rows since symmetric)
    D = np.diag(degrees) # diagonal matrix of degrees
    L = D - W # laplacian matrix
    return D,L

'''
def full_eigendecomp(L):
    full_e_vals, full_e_vectors = npl.eigh(L)
    full_e_inv = npl.inv(full_e_vectors)
    return full_e_vectors, full_e_vals, full_e_inv
'''

def kmeans(L_k, num_clusters, iterations=100, seed=46): # for usage in spectral clustering (feel free to change seed value)
    np.random.seed(seed)
    n, d =  L_k.shape # (kth approximation of L) R^{}
    index = np.random.choice(n, size=num_clusters, replace=False)
    centroids = L_k[index, :]
    for i in range(iterations):
        labels = np.array([
            np.argmin([np.linalg.norm(u - c) for c in centroids]) for u in L_k # min distance of column vector from centroid
        ])
        # update
        new_centroids = np.array([L_k[labels == i].mean(axis=0) if np.any(labels == i) else centroids[i]
                                for i in range(num_clusters)])
        if np.allclose(centroids, new_centroids): # if model converges
            break
        centers = new_centroids
    return labels, centers



chunk_dir = "project/processed_chunks"   # 
sample_rate = 8000

all_features = []
for i in range(1,80):
    fname = f"audio_data_{i:02d}_features.mat"  # zero-padded
    fpath = os.path.join(chunk_dir, fname)
    if os.path.exists(fpath):
        data = io.loadmat(fpath)
        if "feature_vector" in data:
            features = data["feature_vector"]  # shape: (instances, features)
            all_features.append(features)
            print(f"Loaded {fname} with shape {features.shape}")
    else:
        print(f"Skipping {fname} (not found)")


feature_matrix = np.vstack(all_features)  # R^{instances, features}


n_components = 10
scaler = StandardScaler() #

X_scaled = scaler.fit_transform(feature_matrix)


pca = PCA(n_components, random_state=46)

X_pca = pca.fit_transform(X_scaled)
loadings = pca.components_  # R^{n_components, n_features}  # weights of pca


W = cos_affinity(feature_matrix)# 7997 x 7997 map from instances
D, L = laplacian_matr(W)
#print(L.shape) 7997 x 7997 
#print(D.shape) 7997 x 7997 

laplacain_pca = PCA(n_components)
fit_laplacain_pca = laplacain_pca.fit_transform(L)

n_clusters = 5

weights = pca.components_
for i, comp in enumerate(weights):
    print(f"PC{i+1}: {comp}")




 

X_norm = X_pca / np.linalg.norm(X_pca, axis=1, keepdims=True)
kmeans_feat = KMeans(n_clusters=n_clusters, random_state=46, n_init=10)
labels_feat = kmeans_feat.fit_predict(X_norm)

kmeans_lap = KMeans(n_clusters=n_clusters, random_state=46, n_init=10)
labels_lap = kmeans_lap.fit_predict(fit_laplacain_pca)


df_feat = pd.DataFrame(X_pca[:, :2], columns=["PC1", "PC2"])
df_feat["cluster"] = labels_feat

df_lap = pd.DataFrame(fit_laplacain_pca[:, :2], columns=["PC1", "PC2"])
df_lap["cluster"] = labels_lap


plt.figure(figsize=(12,5))

# Feature PCA
plt.subplot(1,2,1)
plt.scatter(df_feat["PC1"], df_feat["PC2"], c=df_feat["cluster"], cmap="tab10", s=20, alpha=0.7)
plt.scatter(kmeans_feat.cluster_centers_[:,0], kmeans_feat.cluster_centers_[:,1],
            c="black", marker="X", s=200, label="Centroids")
plt.title("Feature PCA + Cosine KMeans")
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.legend()


plt.subplot(1,2,2)
plt.scatter(df_lap["PC1"], df_lap["PC2"], c=df_lap["cluster"], cmap="tab10", s=20, alpha=0.7)
plt.scatter(kmeans_lap.cluster_centers_[:,0], kmeans_lap.cluster_centers_[:,1],
            c="black", marker="X", s=200, label="Centroids")
plt.title("Laplacian PCA + KMeans")
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.legend()

plt.tight_layout()
plt.show()




