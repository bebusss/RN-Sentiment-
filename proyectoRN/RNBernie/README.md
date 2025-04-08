# Red Neuronal para Análisis de Sentimientos en Reseñas IMDB

Este proyecto implementa una red neuronal feed-forward para clasificar reseñas de películas de IMDB como positivas o negativas.

## Estructura del Proyecto

```
RNBernie/
├── README.md
├── requirements.txt
├── neural_network.py     # Implementación de la red neuronal
├── imdb_loader.py       # Preprocesamiento y carga de datos
└── main.py             # Script principal de entrenamiento
```

## Arquitectura de la Red Neuronal

- **Capa de entrada**: 5000 características (vectores TF-IDF del texto)
- **Capas ocultas**: 3 capas [128, 64, 32] neuronas con activación ReLU
- **Capa de salida**: 1 neurona con activación sigmoid
- **Inicialización**: Xavier/Glorot para mejor convergencia
- **Función de pérdida**: Binary Cross-Entropy
- **Optimizador**: Mini-batch Gradient Descent (batch_size=64)
- **Learning rate**: 0.001

## Preprocesamiento de Texto

1. Conversión a minúsculas
2. Eliminación de etiquetas HTML
3. Eliminación de caracteres especiales y números
4. Vectorización usando TF-IDF (5000 características)

## Resultados

- **Precisión en test**: 87.54%
- **Loss final**: 0.0415
- **Épocas de entrenamiento**: 50
- **Tiempo aproximado de entrenamiento**: ~10-15 minutos

## Configuración del Entorno

### Requisitos del Sistema
- Python 3.8 o superior
- pip (gestor de paquetes de Python)

### Instalación de Dependencias

```bash
# Crear un entorno virtual (opcional pero recomendado)
python -m venv venv

# Activar el entorno virtual
# En Windows:
venv\Scripts\activate
# En macOS/Linux:
source venv/bin/activate

# Instalar dependencias
pip install numpy pandas scikit-learn matplotlib

# O instalar todas las dependencias desde requirements.txt
pip install -r requirements.txt
```

### Dataset
El dataset debe estar en formato CSV con las columnas 'review' y 'sentiment'.
Colocar el archivo 'IMDB Dataset.csv' en la carpeta RNBernie/.

## Ejecución

```bash
python main.py
```

## Detalles de Implementación

### Características Clave
1. **Prevención de Overflow**: Clipping en la función sigmoid
2. **Mini-batch Processing**: Mejora la eficiencia y estabilidad
3. **Inicialización Xavier**: Previene la desaparición/explosión del gradiente

### Métricas y Visualización
- Gráfica de pérdida durante el entrenamiento
- Precisión en conjunto de prueba
- Ejemplos de predicciones

## Resultados y Análisis

La red neuronal muestra un buen rendimiento en la clasificación de sentimientos:

1. **Curva de Aprendizaje**:
   - Convergencia estable
   - Reducción consistente de la pérdida
   - Sin sobreajuste aparente

2. **Precisión**:
   - 87.54% en datos de prueba
   - Significativamente mejor que el baseline (50%)

## Posibles Mejoras Futuras

1. Implementar validación cruzada
2. Agregar regularización (dropout, L2)
3. Experimentar con diferentes arquitecturas
4. Implementar early stopping
5. Aumentar el vocabulario TF-IDF

## Comandos Rápidos de Referencia

```bash
# Instalación de dependencias individuales
pip install numpy
pip install pandas
pip install scikit-learn
pip install matplotlib

# O instalar todo junto
pip install -r requirements.txt

# Ejecutar el entrenamiento
python main.py
``` 