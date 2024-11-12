# Commented out IPython magic to ensure Python compatibility.
import numpy as py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

import warnings

warnings.filterwarnings('ignore')

data = '/content/wine-clustering.csv'

df = pd.read_csv(data)

df.shape
df.head()

df.info()

df.describe()

df['Alcohol'].unique()

len(df['Alcohol'].unique())

X = df
y = df['Alcohol']

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

X['Alcohol'] = le.fit_transform(X['Alcohol'])

y = le.transform(y)

X.info()

X.head()

cols = X.columns

from sklearn.preprocessing import MinMaxScaler

ms = MinMaxScaler()

X = ms.fit_transform(X)

X = pd.DataFrame(X, columns=[cols])

X.head()

#K-Means model with two clusters

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2, random_state=0)

kmeans.fit(X)

kmeans.cluster_centers_

#Inertia

kmeans.inertia_

labels = kmeans.labels_

# check how many of the samples were correctly labeled
correct_labels = sum(y == labels)

print("Result: %d out of %d samples were correctly labeled." % (correct_labels, y.size))

print('Accuracy score: {0:0.2f}'. format(correct_labels/float(y.size)))

from sklearn.cluster import KMeans
cs = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(X)
    cs.append(kmeans.inertia_)
plt.plot(range(1, 11), cs)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('CS')
plt.show()

kmeans = KMeans(n_clusters=3, random_state=0)

kmeans.fit(X)

# check how many of the samples were correctly labeled
labels = kmeans.labels_

correct_labels = sum(y == labels)

print("Result: %d out of %d samples were correctly labeled." % (correct_labels, y.size))
print('Accuracy score: {0:0.2f}'. format(correct_labels/float(y.size)))

# Convert X to a NumPy array
X = X.values

# Plotting clusters and centroids
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

plt.figure(figsize=(8, 6))
plt.scatter(X[labels == 0][:, 0], X[labels == 0][:, 1], s=50, c='blue', label='Cluster 1')
plt.scatter(X[labels == 1][:, 0], X[labels == 1][:, 1], s=50, c='red', label='Cluster 2')
plt.scatter(X[labels == 2][:, 0], X[labels == 2][:, 1], s=50, c='green', label='Cluster 3')
plt.scatter(centroids[:, 0], centroids[:, 1], s=200, c='black', marker='*', label='Centroids')
plt.title("Kmeans Iteration 1")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()
