import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots
import random

from ISLP import load_data
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import mean_absolute_error
from scipy.cluster.hierarchy import dendrogram, cut_tree
from ISLP.cluster import compute_linkage

weather = load_data('Weather') #Load the weather dataset

# Select features for clustering (excluding TrackTemp)
# Because we want to predict it from the other features listed, or it will be cheating.
feature = [
    "AirTemp",
    "Humidity",
    "Pressure",
    "Rainfall",
    "WindSpeed",
]

# Create feature matrix X
X = weather[feature]

# Standardize the features so all variables are on the same scale
# Important for K-Means bc it uses Euclidean distance
scaler = StandardScaler().set_output(transform="pandas")
X_scaled = scaler.fit_transform(X)

# Apply K-Means clustering
# Want 4 clusters in this case for example (n_clusters)
# random_state for reproducibility (same results every run)
kmeans = KMeans(n_clusters=4, random_state=42)

# Assigning each data point to a cluester
weather['cluster'] = kmeans.fit_predict(X_scaled)

# Compute average TrackTemp for each cluster
# Helps understand how TrackTemp varies across clusters
cluster_temp_map = weather.groupby('cluster')['TrackTemp'].mean()

# Map each row's cluster to its corresponding average TrackTemp
# This acts as a simple way to estimate TrackTemp from clusters
weather['predicted_TrackTemp'] = weather['cluster'].map(cluster_temp_map)

# Evaluate how close our estimated TrackTemp is to the actual TrackTemp
# Note: clustering is unsupervised, so this is just extra analysis
mae = mean_absolute_error(weather['TrackTemp'], weather['predicted_TrackTemp'])
print("Mean Absolute Error:", mae)

# Visualize clusters using AirTemp and TrackTemp
# Points are colored based on their cluster assignment
plt.scatter(weather['AirTemp'], weather['TrackTemp'], c=weather['cluster'])

# Label axes and add title
plt.xlabel('AirTemp')
plt.ylabel('TrackTemp')
plt.title('K-Means Clustering of Weather Data on Track Temperature')

# Colorbar to show cluster labels
plt.colorbar(label='Cluster')

#Display
plt.show()