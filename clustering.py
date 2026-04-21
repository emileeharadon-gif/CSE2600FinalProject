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

weather = load_data('Weather')

feature = [
    "AirTemp",
    "Humidity",
    "Pressure",
    "Rainfall",
    "WindSpeed",
]

X = weather[feature]
scaler = StandardScaler().set_output(transform="pandas")
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=4, random_state=42)
weather['cluster'] = kmeans.fit_predict(X_scaled)

cluster_temp_map = weather.groupby('cluster')['TrackTemp'].mean()
weather['predicted_TrackTemp'] = weather['cluster'].map(cluster_temp_map)

mae = mean_absolute_error(weather['TrackTemp'], weather['predicted_TrackTemp'])
print("Mean Absolute Error:", mae)

plt.scatter(weather['AirTemp'], weather['TrackTemp'], c=weather['cluster'])
plt.xlabel('AirTemp')
plt.ylabel('TrackTemp')
plt.title('K-Means Clustering of Weather Data on Track Temperature')
plt.colorbar(label='Cluster')
plt.show()