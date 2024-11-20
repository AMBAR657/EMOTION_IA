from flask import Flask, render_template, send_from_directory
import os
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Crear la aplicación Flask
app = Flask(__name__)

# Ruta para la página principal
@app.route('/')
def home():
    return render_template('index.html')

# Ruta para los gráficos
@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

# Generar el modelo y gráficos al iniciar
def generate_model_and_graphs():
    # Crear el DataSet
    data = {
        "Feature1": np.random.rand(100),
        "Feature2": np.random.rand(100),
        "Feature3": np.random.rand(100),
        "Output": np.random.choice(["Class1", "Class2"], size=100),
    }
    df = pd.DataFrame(data)

    # Transformar la variable de salida a numérica
    df["Output"] = df["Output"].map({"Class1": 0, "Class2": 1})

    # Dividir el DataSet
    X = df[["Feature1", "Feature2", "Feature3"]]
    y = df["Output"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Reducir el número de atributos
    X_reduced = X[["Feature1", "Feature2"]]
    X_train_red, X_test_red, y_train_red, y_test_red = train_test_split(
        X_reduced, y, test_size=0.3, random_state=42
    )

    # Entrenar el modelo
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train_red, y_train_red)

    # Graficar el árbol de decisión
    plt.figure(figsize=(10, 6))
    plot_tree(model, feature_names=X_reduced.columns, class_names=["Class1", "Class2"], filled=True)
    tree_path = os.path.join('static', 'decision_tree.png')
    plt.savefig(tree_path)
    plt.close()

    # Graficar el límite de decisión
    x_min, x_max = X_train_red["Feature1"].min() - 1, X_train_red["Feature1"].max() + 1
    y_min, y_max = X_train_red["Feature2"].min() - 1, X_train_red["Feature2"].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Paired)
    plt.scatter(X_train_red["Feature1"], X_train_red["Feature2"], c=y_train_red, edgecolor="k", cmap=plt.cm.Paired)
    plt.title("Límite de decisión")
    plt.xlabel("Feature1")
    plt.ylabel("Feature2")
    boundary_path = os.path.join('static', 'decision_boundary.png')
    plt.savefig(boundary_path)
    plt.close()

# Generar gráficos al iniciar la aplicación
generate_model_and_graphs()

if __name__ == '__main__':
    app.run(debug=True)
