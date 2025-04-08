"""
Implementación de una Red Neuronal Feed-Forward para clasificación binaria.
Esta red neuronal está diseñada específicamente para el análisis de sentimientos,
utilizando múltiples capas ocultas con activación ReLU y una capa de salida con sigmoid.
"""

import numpy as np
from typing import List, Tuple

class NeuralNetwork:
    def __init__(self, layer_sizes: List[int], learning_rate: float = 0.01):
        """
        Inicializa la red neuronal con las capas especificadas.
        
        Args:
            layer_sizes: Lista con el tamaño de cada capa. Por ejemplo: [5000, 128, 64, 32, 1]
                        - Primera capa (5000): Tamaño de entrada (características TF-IDF)
                        - Capas intermedias (128, 64, 32): Capas ocultas
                        - Última capa (1): Capa de salida para clasificación binaria
            learning_rate: Tasa de aprendizaje para gradient descent
        """
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.weights = []    # Pesos entre capas
        self.biases = []     # Términos de sesgo para cada capa
        self.activations = [] # Valores de activación para cada capa
        self.z_values = []    # Valores pre-activación para cada capa
        
        # Inicialización de pesos usando Xavier/Glorot
        # Esto ayuda a mantener la varianza de las activaciones similar entre capas
        for i in range(len(layer_sizes) - 1):
            scale = np.sqrt(2.0 / (layer_sizes[i] + layer_sizes[i + 1]))
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * scale)
            self.biases.append(np.zeros((1, layer_sizes[i + 1])))
    
    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        Función de activación sigmoid con estabilidad numérica.
        f(x) = 1 / (1 + e^(-x))
        
        Args:
            z: Entrada a la función sigmoid
        
        Returns:
            Valores activados entre 0 y 1
        """
        # Clippeamos los valores para evitar overflow numérico
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def sigmoid_derivative(self, z: np.ndarray) -> np.ndarray:
        """
        Derivada de la función sigmoid.
        f'(x) = f(x) * (1 - f(x))
        
        Args:
            z: Valores de activación sigmoid
        
        Returns:
            Derivada de la función sigmoid
        """
        return z * (1 - z)
    
    def relu(self, z: np.ndarray) -> np.ndarray:
        """
        Función de activación ReLU (Rectified Linear Unit).
        f(x) = max(0, x)
        
        Args:
            z: Entrada a la función ReLU
        
        Returns:
            Valores activados (0 para entradas negativas, x para positivas)
        """
        return np.maximum(0, z)
    
    def relu_derivative(self, z: np.ndarray) -> np.ndarray:
        """
        Derivada de la función ReLU.
        f'(x) = 1 si x > 0, 0 en otro caso
        
        Args:
            z: Valores de activación ReLU
        
        Returns:
            Derivada de la función ReLU
        """
        return np.where(z > 0, 1, 0)
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Propagación hacia adelante (forward propagation).
        Calcula la salida de la red para una entrada dada.
        
        Args:
            X: Datos de entrada de forma (n_samples, n_features)
        
        Returns:
            Predicciones de la red (valores entre 0 y 1)
        """
        self.activations = [X]
        self.z_values = []
        
        # Iteramos por cada capa de la red
        for i in range(len(self.weights)):
            # Calculamos la combinación lineal: z = X·W + b
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            
            # Aplicamos la función de activación
            if i == len(self.weights) - 1:
                # Capa de salida: usamos sigmoid para clasificación binaria
                activation = self.sigmoid(z)
            else:
                # Capas ocultas: usamos ReLU
                activation = self.relu(z)
            
            self.activations.append(activation)
        
        return self.activations[-1]
    
    def backward(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Retropropagación (backward propagation).
        Calcula los gradientes y actualiza los pesos y sesgos.
        
        Args:
            X: Datos de entrada
            y: Etiquetas verdaderas
        """
        m = X.shape[0]  # Número de ejemplos
        delta = self.activations[-1] - y  # Error en la capa de salida
        
        # Iteramos por las capas en orden inverso
        for i in range(len(self.weights) - 1, -1, -1):
            # Calculamos los gradientes
            dW = np.dot(self.activations[i].T, delta)  # Gradiente de los pesos
            db = np.sum(delta, axis=0, keepdims=True)  # Gradiente de los sesgos
            
            if i > 0:
                # Propagamos el error a la capa anterior
                delta = np.dot(delta, self.weights[i].T) * self.relu_derivative(self.activations[i])
            
            # Actualizamos pesos y sesgos usando gradient descent
            self.weights[i] -= self.learning_rate * dW
            self.biases[i] -= self.learning_rate * db
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int, batch_size: int = 32) -> List[float]:
        """
        Entrena la red neuronal usando mini-batch gradient descent.
        
        Args:
            X: Datos de entrenamiento
            y: Etiquetas de entrenamiento
            epochs: Número de épocas de entrenamiento
            batch_size: Tamaño de los mini-batches
        
        Returns:
            Lista con los valores de pérdida por época
        """
        losses = []
        n_samples = X.shape[0]
        n_batches = n_samples // batch_size
        
        for epoch in range(epochs):
            epoch_loss = 0
            
            # Mezclamos los datos en cada época
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            # Entrenamiento por mini-batches
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size
                
                # Seleccionamos el mini-batch actual
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # Forward pass
                predictions = self.forward(X_batch)
                
                # Calculamos la pérdida (binary cross-entropy)
                batch_loss = np.mean(-y_batch * np.log(predictions + 1e-15) - 
                                   (1 - y_batch) * np.log(1 - predictions + 1e-15))
                epoch_loss += batch_loss
                
                # Backward pass
                self.backward(X_batch, y_batch)
            
            # Promediamos la pérdida de la época
            epoch_loss /= n_batches
            losses.append(epoch_loss)
            
            # Mostramos el progreso cada 10 épocas
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")
        
        return losses
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Realiza predicciones sobre nuevos datos.
        
        Args:
            X: Datos de entrada
        
        Returns:
            Predicciones binarias (0 o 1)
        """
        predictions = self.forward(X)
        return (predictions > 0.5).astype(int) 