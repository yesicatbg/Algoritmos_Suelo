import networkx as nx
import pandas as pd
from geopy.distance import geodesic
from itertools import combinations
import matplotlib.pyplot as plt

# Leer los datos desde un archivo CSV u otro origen de datos
data = pd.read_excel('Kmeans- prueba small.xlsx')

# Crear un grafo vacío
G = nx.Graph()

# Iterar sobre cada fila de los datos y agregar los nodos al grafo
for index, row in data.iterrows():
    # Obtener los valores de las columnas
    temperatura = row['Temperaturasht Normalizada']
    humedad = row['Humedad sht normalizada']
    temperatura_ambiente = row['Temperatura BME normalizada']
    latitud = row['Latitud']
    longitud = row['Longitud']
  
    # Crear un nodo con los valores de temperatura, humedad y temperatura ambiente
    nodo = (temperatura, humedad, temperatura_ambiente)
    
    # Agregar el nodo al grafo y asignarle atributos adicionales como latitud y longitud
    G.add_node(nodo, latitud=latitud, longitud=longitud)

# Calcular las distancias geodésicas entre todos los pares de nodos
for node1, node2 in combinations(G.nodes(), 2):
    latitud1 = G.nodes[node1]['latitud']
    longitud1 = G.nodes[node1]['longitud']
    latitud2 = G.nodes[node2]['latitud']
    longitud2 = G.nodes[node2]['longitud']
    
    distancia = geodesic((latitud1, longitud1), (latitud2, longitud2)).kilometers
    
    # Establecer una conexión si los nodos están dentro de un umbral de distancia específico
    umbral_distancia = 0.0002546715823998821
    if distancia <= umbral_distancia:
        G.add_edge(node1, node2)


# Ejemplo: Imprimir el número de nodos y aristas del grafo
print("Número de nodos:", G.number_of_nodes())
print("Número de aristas:", G.number_of_edges())

nx.draw(G, with_labels=True)
plt.show()