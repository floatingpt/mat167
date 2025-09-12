import numpy as np
import os
import numpy.linalg as npl
import scipy.io as io
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

'''

j_i is a feature vector for the ith audio file j_i \in R^{1 x 129} (129 is step choice) (accidentally added zeroth index on for loop)

use cosine kernel to compute ijth value of weight matrix 

x_i^T x_j / ||x_i|| ||x_kj|| = cos(theta)

map negative values to zero


FINAL_NOTE ----- THIS FILE IS not complete!! I need to map the clusters to their physical features via the pca weights

'''

'''
def cos_kernel(feature_matrix): # input feature vectors and perfrom (ab^T/ ||a|| ||b||)
    num = feature_matrix.T @ feature_matrix # inner product
    denom = npl.norm(feature_matrix)
    denom = denom**2
    return num/denom 
'''

def make_feature_names(n_mfcc=13):
    names = []
    #mfcc mean
    for i in range(1, n_mfcc+1):
        names.append(f"mfcc_mean_{i}")
    # mfcc variance
    for i in range(1, n_mfcc+1):
        names.append(f"mfcc_var_{i}")
    # feature names
    names += ["centroid", "bandwidth", "rolloff", "flux"]
    return names

feature_names= make_feature_names() # 2 * 13 + 4 
# total of 30 features 



def cos_affinity(feature_matrix): 
    feature_matrix_n = normalize(feature_matrix, axis =1) # turn into unit vector by dividing by norm of column vector
    W = cosine_similarity(feature_matrix_n) # output is unit vector 
    W[W < 0] = 0 # map all negative valyes to zero
    np.fill_diagonal(W, 0) # items should not have affinity to themselves
    W = (W + W.T) / 2
    return W


def laplacian_matr(W):  #  L = D - W
    degrees = np.sum(W, axis=1) # sum over cols (equal to sum over rows since symmetric)
    D = np.diag(degrees) # diagonal matrix of degrees
    L = D - W # laplacian matrix
    return D,L

# not needed - can use eigh() instead 
''' def full_eigendecomp(L):
    full_e_vals, full_e_vectors = npl.eigh(L)
    full_e_inv = npl.inv(full_e_vectors)
    return full_e_vectors, full_e_vals, full_e_inv
'''

 # use sklearn method instead 
''' def kmeans(L_k, num_clusters, iterations=100, seed=46): # for usage in spectral clustering (feel free to change seed value)
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
'''



chunk_dir = "processed_chunks"   # change to relative path name on your local environment
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

# pca 

def do_pca(feature_matrix, n_components = 10):

    scaler = StandardScaler() #
    X_scaled = scaler.fit_transform(feature_matrix)
    pca = PCA(n_components, random_state=46)
    X_pca = pca.fit_transform(X_scaled)
    kmeans = KMeans(n_clusters=5, random_state=46, n_init=10)
    labels = kmeans.fit_predict(X_pca)
    return scaler, pca, X_scaled, X_pca, kmeans, labels

    #X_pca = pca.fit_transform(X_scaled)
    #loadings = pca.components_  # R^{n_components, n_features}  # weights of pca




W = cos_affinity(feature_matrix)# 7997 x 7997 map from instances
D, L = laplacian_matr(W)
    #print(L.shape) 7997 x 7997 
    #print(D.shape) 7997 x 7997 
eig_vals_L, eig_vect_L = np.linalg.eigh(L) # find eigen values and vectors 



n_clusters = 5

scaler_ft, pca_ft, X_scaled_ft, X_pca_ft, kmeans_ft, labels_ft= do_pca(feature_matrix, n_components=2)
scaler_L, pca_L, X_scaled_L, X_pca_L, kmeans_L, labels_L = do_pca(L, n_components=2)


# 6) interpret clusters: per-cluster average features, top contributing features per PC, features that separate clusters
def summarize_clusters(feature_matrix, labels, feature_names, top_k=5):

    df = pd.DataFrame(feature_matrix, columns=feature_names) # make data frame
    df["cluster"] = labels # add labeling
    cluster_means = df.groupby("cluster").mean()# add by means
    global_mean = df.drop(columns="cluster").mean()
    summary = {}
    # for each cluster, compute how far each feature is from the global mean (absolute or signed)
    for cl in sorted(df["cluster"].unique()):
        means = df[df.cluster == cl].drop(columns="cluster").mean() # mean by cluster
        #global_mn = df.drop(columns="cluster").mean # overall 
        diffs = means -  global_mean
        # sort by absolute difference but keep sign
        top = diffs.abs().sort_values(ascending=False).head(5) # top 5 in df
        summary[cl] = [(feat, diffs[feat]) for feat in top.index]
    return summary



summary = summarize_clusters(feature_matrix, labels_ft, feature_names)

print("\nCluster interpretation:")
for cl, feats in summary.items():
    print(f"Cluster {cl}:")
    for f, diff in feats:
        print(f"   {f}: {diff:.3f}")



plt.figure(figsize=(12,5))

# Feature PCA
plt.subplot(1,2,1)
plt.scatter(X_pca_ft[:,0], X_pca_ft[:,1], c=labels_ft, cmap="tab10", s=20, alpha=0.7)
plt.scatter(kmeans_ft.cluster_centers_[:,0], kmeans_ft.cluster_centers_[:,1],
            c="black", marker="X", s=200, label="Centroids")
plt.title("Feature PCA + Cosine KMeans")
plt.xlabel("PC1"); plt.ylabel("PC2"); plt.legend()

# Laplacian PCA
plt.subplot(1,2,2)
plt.scatter(X_pca_L[:,0], X_pca_L[:,1], c=labels_L, cmap="tab10", s=20, alpha=0.7)
plt.scatter(kmeans_L.cluster_centers_[:,0], kmeans_L.cluster_centers_[:,1],
            c="black", marker="X", s=200, label="Centroids")
plt.title("Laplacian PCA + KMeans")
plt.xlabel("PC1"); plt.ylabel("PC2"); plt.legend()

plt.tight_layout()
plt.show()