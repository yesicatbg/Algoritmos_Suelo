from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.tree import export_text
from sklearn.metrics import classification_report, confusion_matrix


# Leer el archivo Excel
df = pd.read_excel('Kmeans- tree.xlsx', sheet_name='Kmeans')

# Verificar los datos leídos
print(df.head())

# Seleccionar las columnas de características
X = df[['Temperaturasht Normalizada', 'Humedad sht normalizada', 'Temperatura BME normalizada']]

# Seleccionar la columna objetivo
y = df['Etiquetas']

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear un clasificador de árbol de decisión
clf = DecisionTreeClassifier()

# Entrenar el clasificador utilizando los datos de entrenamiento
clf.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = clf.predict(X_test)

# Calcular la precisión del clasificador
accuracy = accuracy_score(y_test, y_pred)
print("Precision:", accuracy)

# Precisión y rendimiento
print("Matriz de confusión:")
print(confusion_matrix(y_test, y_pred))
print("Informe de clasificación:")
print(classification_report(y_test, y_pred))

# Calcular la profundidad del árbol
profundidad = clf.get_depth()
print("Profundidad del árbol:", profundidad)

# Calcular número de nodos y hojas
num_nodos = clf.tree_.node_count
num_hojas = clf.tree_.n_leaves
print("Número de nodos:", num_nodos)
print("Número de hojas:", num_hojas)

# Caracteristica mas importante
importances = clf.feature_importances_
for i, feature_name in enumerate(X.columns):
    print("Importancia de", feature_name, ":", importances[i])

# Regla de decisión
tree_rules = export_text(clf, feature_names=X.columns.tolist())
print(tree_rules)


# Graficar el árbol de decisión
plt.figure(figsize=(16,16))
plot_tree(clf, filled=True, rounded=True, feature_names=X.columns, class_names=True)
plt.show()

