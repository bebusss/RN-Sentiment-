"""
Script principal para entrenar y evaluar la red neuronal en el dataset IMDB.
Este script:
1. Carga y preprocesa el dataset
2. Configura y entrena la red neuronal
3. Evalúa el rendimiento del modelo
4. Visualiza los resultados
"""

import numpy as np
import matplotlib.pyplot as plt
from neural_network import NeuralNetwork
from imdb_loader import IMDBDataLoader
import os

def plot_loss(losses):
    """
    Genera y guarda una gráfica de la pérdida durante el entrenamiento.
    
    La gráfica muestra cómo la función de pérdida (binary cross-entropy)
    disminuye a lo largo de las épocas, lo que indica el aprendizaje del modelo.
    
    Args:
        losses: Lista de valores de pérdida por época
    """
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Training Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('training_loss.png')
    plt.close()

def main():
    """
    Función principal que ejecuta el pipeline completo de entrenamiento.
    
    Pasos:
    1. Carga y preprocesamiento de datos
    2. Configuración de la red neuronal
    3. Entrenamiento del modelo
    4. Evaluación y visualización de resultados
    """
    # Obtener la ruta al archivo del dataset
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(current_dir, 'IMDB Dataset.csv')
    
    # Inicializar el cargador de datos
    loader = IMDBDataLoader()
    
    # Cargar y preprocesar los datos
    print("Cargando dataset IMDB...")
    X_train, X_test, y_train, y_test = loader.load_data(dataset_path)
    
    # Reshape de las etiquetas para coincidir con el formato esperado
    y_train = y_train.reshape(-1, 1)  # Convertir a columna
    y_test = y_test.reshape(-1, 1)
    
    # Definir la arquitectura de la red neuronal
    input_size = X_train.shape[1]  # Número de características TF-IDF
    # Arquitectura: [entrada, capa1, capa2, capa3, salida]
    layer_sizes = [input_size, 128, 64, 32, 1]  # 3 capas ocultas
    
    # Inicializar la red neuronal
    print("\nInicializando red neuronal...")
    nn = NeuralNetwork(
        layer_sizes=layer_sizes,
        learning_rate=0.001  # Tasa de aprendizaje pequeña para estabilidad
    )
    
    # Entrenar la red neuronal
    print("\nIniciando entrenamiento...")
    losses = nn.train(
        X_train,
        y_train,
        epochs=50,      # Número de épocas de entrenamiento
        batch_size=64   # Tamaño de mini-batch para gradient descent
    )
    
    # Generar y guardar la gráfica de pérdida
    plot_loss(losses)
    
    # Evaluar el modelo en el conjunto de prueba
    print("\nEvaluando modelo...")
    predictions = nn.predict(X_test)
    accuracy = np.mean(predictions == y_test)
    print(f"Precisión en test: {accuracy:.4f}")
    
    # Mostrar ejemplos de predicciones
    print("\nEjemplos de predicciones:")
    for i in range(5):
        print(f"Etiqueta verdadera: {y_test[i][0]}, Predicción: {predictions[i][0]}")

if __name__ == "__main__":
    main() 