import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# --- scikit-learn API ---

# Q1
years = np.array([1, 2, 3, 5, 7, 10]).reshape(-1, 1)
salary = np.array([45000, 50000, 60000, 75000, 90000, 120000])

model = LinearRegression()
model.fit(years, salary)

pred4 = model.predict([[4]])
pred8 = model.predict([[8]])

print(f"Slope: {model.coef_[0]}")
print(f"Intercept: {model.intercept_}")
print(f"Prediction for 4 years: {pred4[0]}")
print(f"Prediction for 8 years: {pred8[0]}")

# Q2
x = np.array([10, 20, 30, 40, 50])
print(f"Original shape: {x.shape}")
x_2d = x.reshape(-1, 1)
print(f"New shape: {x_2d.shape}")
# scikit-learn needs X to be 2D because it treats the input as a matrix where each row is a sample and each column is a feature, allowing for multiple features per sample.

# Q3
X_clusters, _ = make_blobs(n_samples=120, centers=3, cluster_std=0.8, random_state=7)

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_clusters)
labels = kmeans.predict(X_clusters)

print(f"Cluster centers: {kmeans.cluster_centers_}")
counts = np.bincount(labels)
print(f"Points in cluster 0: {counts[0]}")
print(f"Points in cluster 1: {counts[1]}")
print(f"Points in cluster 2: {counts[2]}")

plt.scatter(X_clusters[:, 0], X_clusters[:, 1], c=labels, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', s=200, c='black')
plt.title('K-Means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.savefig('outputs/kmeans_clusters.png')