# -*- coding: utf-8 -*-
"""Activity1: K-means Clustering"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

# Load dataset (Breast Cancer data set from Kaggle)
data = pd.read_csv('data.csv')
print(data.head())

# Drop unnecessary columns
if 'id' in data.columns:
    data.drop(columns=['id'], inplace=True)

if 'diagnosis' in data.columns:
    true_labels = LabelEncoder().fit_transform(data['diagnosis'])
    data.drop(columns=['diagnosis'], inplace=True)
else:
    true_labels = None

# Impute missing values using the mean strategy
imputer = SimpleImputer(strategy='mean') # Create an imputer instance
data_imputed = imputer.fit_transform(data) # Impute missing values

# Standardize data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_imputed) # Use imputed data for scaling

print(data.head())

# Elbow Method
inertia = []
k_range = range(2, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data_scaled)
    inertia.append(kmeans.inertia_)

    # Plot Elbow Method
plt.figure(figsize=(8, 5))
plt.plot(k_range, inertia, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.show()

# Optimal K selection
optimal_k = 4

# K-Means Clustering
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
cluster_labels = kmeans.fit_predict(data_scaled)
centroids = kmeans.cluster_centers_

# Visualize Clusters (using PCA for 2D visualization)
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_scaled)
centroids_pca = pca.transform(centroids)

plt.figure(figsize=(8, 5))
for cluster in range(optimal_k):
    plt.scatter(data_pca[cluster_labels == cluster, 0],
                data_pca[cluster_labels == cluster, 1],
                label=f"Cluster {cluster}")
plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1],
            s=300, c='red', label='Centroids', marker='X')
plt.title("Clusters Visualization (PCA Reduced)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.show()

# Silhouette Coefficient
silhouette_avg = silhouette_score(data_scaled, cluster_labels)
print(f"Silhouette Coefficient: {silhouette_avg:.2f}")

# Adjusted Rand Index
if true_labels is not None:
    ari_score = adjusted_rand_score(true_labels, cluster_labels)
    print(f"Adjusted Rand Index: {ari_score:.2f}")
else:
    print("True labels not provided, ARI cannot be computed.")

"""The k-means clustering on the Breast Cancer dataset with an optimal cluster count of 4 produced moderate results. The silhouette coefficient of 0.27 indicates that the clusters have some overlap and are not well-separated, suggesting that the algorithm's separation of data points could be improved. However, the adjusted Rand index of 0.60 shows a moderate agreement with the true labels, indicating that the clustering somewhat aligns with the actual classification. While the k-means algorithm provided valuable insights, further refinement in feature selection or clustering approach may enhance its effectiveness.

"""

for i, ax in enumerate(axes.flatten()):
    ax.scatter(x=df_d[cols[i]], y=df_d['target'])
    ax.set_title(str(cols[i]))

fig.text(0.1, 0.5, 'Diabetes Progression', va='center', rotation='vertical')
