from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import silhouette_score
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

# Especifica la ruta del archivo Excel
ruta_archivo = r'C:\Users\yesic\OneDrive - Pontificia Universidad Javeriana\Documents\Kmeans.xlsx'

# Lee el archivo Excel en un DataFrame
datos = pd.read_excel(ruta_archivo)

# Configuración de K-means
k = 3  # Número de clusters que deseas obtener
kmeans = KMeans(n_clusters=k, n_init=10)

# Aplicación de K-means
kmeans.fit(datos)

# Obtención de las etiquetas de los clusters asignados a cada muestra de datos
etiquetas = kmeans.labels_

# Obtención de los centroides de los clusters
centroides = kmeans.cluster_centers_

# Calcula la inercia
inercia = kmeans.inertia_
print("Inercia:", inercia)

# Calcula el coeficiente de silueta
silhouette_coef = silhouette_score(datos, kmeans.labels_)
print("Coeficiente de Silueta:", silhouette_coef)

# Calcula la matriz de distancias entre los centroides de los clústeres
centroids = []
for label in np.unique(kmeans.labels_):
    centroid = np.mean(datos[kmeans.labels_ == label], axis=0)
    centroids.append(centroid)
centroids = np.array(centroids)
centroid_distances = euclidean_distances(centroids)

# Calcula la matriz de distancias entre todas las muestras
sample_distances = euclidean_distances(datos)

# Calcula la mínima distancia entre los clústeres
min_cluster_distance = np.min(centroid_distances[centroid_distances > 0])

# Calcula la máxima distancia intra-cluster
max_intracluster_distance = np.max(sample_distances[np.where(kmeans.labels_[:, None] == kmeans.labels_[None, :])])

# Calcula el índice de Dunn
dunn_index = min_cluster_distance / max_intracluster_distance

print("Indice de Dunn:", dunn_index)

# Crear gráfica de dispersión
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(datos['Temperaturasht Normalizada'], datos['Humedad sht normalizada'], datos['Temperatura BME normalizada'], c=etiquetas)
ax.scatter(centroides[:, 0], centroides[:, 1], centroides[:, 2], c='red', marker='x', label='Centroides')
ax.set_xlabel('Temperaturasht Normalizada')
ax.set_ylabel('Humedad sht normalizada')
ax.set_zlabel('Temperatura BME normalizada')
ax.set_title('Gráfica de Dispersión en 3D con K-means')
plt.legend()
plt.show()