"""
Módulo para cargar y preprocesar el dataset IMDB de reseñas de películas.
Este módulo se encarga de:
1. Cargar el dataset desde un archivo CSV
2. Limpiar y preprocesar el texto de las reseñas
3. Convertir el texto en vectores numéricos usando TF-IDF
4. Dividir los datos en conjuntos de entrenamiento y prueba
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import re

class IMDBDataLoader:
    def __init__(self):
        """
        Inicializa el cargador de datos IMDB.
        Configura el vectorizador TF-IDF con un límite de 5000 características
        para mantener el modelo manejable y eficiente.
        """
        # TF-IDF (Term Frequency-Inverse Document Frequency)
        # - Convierte texto en vectores numéricos
        # - Considera tanto la frecuencia de las palabras como su importancia
        # - max_features=5000 limita el vocabulario a las 5000 palabras más relevantes
        self.vectorizer = TfidfVectorizer(max_features=5000)
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocesa el texto de una reseña.
        
        Pasos de preprocesamiento:
        1. Conversión a minúsculas
        2. Eliminación de etiquetas HTML
        3. Eliminación de caracteres especiales y números
        4. Eliminación de espacios extra
        
        Args:
            text: Texto de la reseña
        
        Returns:
            Texto limpio y preprocesado
        """
        # Convertir a minúsculas para normalizar el texto
        text = text.lower()
        
        # Eliminar etiquetas HTML (ej: <br>, <i>, etc.)
        text = re.sub(r'<[^>]+>', '', text)
        
        # Eliminar caracteres especiales y números
        # Solo mantener letras y espacios
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Eliminar espacios múltiples y espacios al inicio/final
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def load_data(self, filepath: str) -> tuple:
        """
        Carga y preprocesa el dataset IMDB.
        
        Proceso:
        1. Carga el CSV
        2. Convierte sentimientos a valores binarios
        3. Preprocesa el texto
        4. Divide en train/test
        5. Vectoriza el texto usando TF-IDF
        
        Args:
            filepath: Ruta al archivo CSV del dataset
        
        Returns:
            Tupla de (X_train, X_test, y_train, y_test) donde:
            - X_train/X_test: Matrices de características TF-IDF
            - y_train/y_test: Vectores de etiquetas (0=negativo, 1=positivo)
        """
        # Cargar el dataset
        print("Cargando dataset...")
        df = pd.read_csv(filepath)
        
        # Convertir sentimientos a valores binarios (positivo=1, negativo=0)
        df['label'] = (df['sentiment'] == 'positive').astype(int)
        
        # Preprocesar el texto de las reseñas
        print("Preprocesando texto...")
        df['processed_text'] = df['review'].apply(self.preprocess_text)
        
        # Dividir en características (X) y etiquetas (y)
        X = df['processed_text']
        y = df['label'].values
        
        # Dividir en conjuntos de entrenamiento y prueba (80% train, 20% test)
        print("Dividiendo en train/test...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Vectorizar el texto usando TF-IDF
        print("Vectorizando texto...")
        # Ajustar el vectorizador solo con datos de entrenamiento para evitar data leakage
        X_train = self.vectorizer.fit_transform(X_train).toarray()
        # Transformar datos de prueba usando el vocabulario aprendido del training
        X_test = self.vectorizer.transform(X_test).toarray()
        
        print(f"Dimensiones del conjunto de entrenamiento: {X_train.shape}")
        print(f"Dimensiones del conjunto de prueba: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test 