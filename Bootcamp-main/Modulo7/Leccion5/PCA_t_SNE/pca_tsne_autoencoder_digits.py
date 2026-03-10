"""
Proyecto: PCA y t-SNE en acción con autoencoder
Dataset: digits (sklearn)
Descripción:
 - Aplica PCA (2D) y genera scatter plot.
 - Aplica t-SNE con dos valores de perplejidad.
 - Aplica un autoencoder para reducción a 2D.
 - Compara visualmente las técnicas.

Instrucciones del autor (Yuri):
 - Explicar los ejercicios paso a paso, numerando los pasos.
 - Usar lenguaje claro y pedagógico.
"""

# ==========================
# 1. Importar librerías
# ==========================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import tensorflow as tf
from tensorflow.keras import layers, Model


# ==========================
# 2. Fijar semillas aleatorias
# ==========================

# Paso 1: fijar semilla para reproducibilidad básica
np.random.seed(42)
tf.random.set_seed(42)


# ==========================
# 3. Cargar y preprocesar el dataset
# ==========================

# Paso 2: cargar el dataset digits
# X: matriz de características (n_muestras x 64)
# y: etiquetas (dígitos 0-9)

digits = load_digits()
X = digits.data
y = digits.target

# Paso 3: escalar los datos con StandardScaler
# Interpretación: cada columna tendrá media 0 y varianza 1.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ==========================
# 4. PCA a 2 componentes
# ==========================

# Paso 4: crear el modelo PCA para 2 componentes principales
pca = PCA(n_components=2, random_state=42)

# Paso 5: ajustar PCA y transformar los datos
X_pca = pca.fit_transform(X_scaled)

# Paso 6: graficar el resultado de PCA
plt.figure(figsize=(6, 5))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', s=15)
plt.colorbar(scatter, label='Dígito')
plt.xlabel('Componente principal 1')
plt.ylabel('Componente principal 2')
plt.title('PCA (2 componentes) - digits')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ==========================
# 5. t-SNE con dos perplejidades
# ==========================

# Paso 7: aplicar t-SNE con perplejidad 30
# Nota: t-SNE es más costoso; reducimos primero con PCA a 30 dims para velocidad.
pca_30 = PCA(n_components=30, random_state=42)
X_pca_30 = pca_30.fit_transform(X_scaled)

perplexities = [30, 50]
X_tsne_results = {}

for perp in perplexities:
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42, init='pca', learning_rate='auto')
    X_tsne = tsne.fit_transform(X_pca_30)
    X_tsne_results[perp] = X_tsne

    plt.figure(figsize=(6, 5))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', s=15)
    plt.colorbar(scatter, label='Dígito')
    plt.xlabel('Dimensión t-SNE 1')
    plt.ylabel('Dimensión t-SNE 2')
    plt.title(f't-SNE (perplejidad = {perp}) - digits')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# ==========================
# 6. Autoencoder para reducción a 2D
# ==========================

# Paso 8: definir dimensiones del autoencoder
input_dim = X_scaled.shape[1]  # 64 features
encoding_dim = 2               # espacio latente 2D

# Paso 9: construir el modelo autoencoder
input_layer = layers.Input(shape=(input_dim,))

# Encoder
encoded = layers.Dense(32, activation='relu')(input_layer)
encoded = layers.Dense(16, activation='relu')(encoded)
latent = layers.Dense(encoding_dim, activation='linear', name='latent_space')(encoded)

# Decoder
decoded = layers.Dense(16, activation='relu')(latent)
decoded = layers.Dense(32, activation='relu')(decoded)
output_layer = layers.Dense(input_dim, activation='sigmoid')(decoded)

# Modelo completo
autoencoder = Model(inputs=input_layer, outputs=output_layer)

# Modelo encoder separado
encoder = Model(inputs=input_layer, outputs=latent)

# Paso 10: compilar el autoencoder
autoencoder.compile(optimizer='adam', loss='mse')

# Paso 11: entrenar el autoencoder
history = autoencoder.fit(
    X_scaled, X_scaled,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Paso 12: obtener la representación 2D desde el encoder
X_auto_2d = encoder.predict(X_scaled)

# Paso 13: graficar el espacio latente del autoencoder
plt.figure(figsize=(6, 5))
scatter = plt.scatter(X_auto_2d[:, 0], X_auto_2d[:, 1], c=y, cmap='tab10', s=15)
plt.colorbar(scatter, label='Dígito')
plt.xlabel('Latent dim 1')
plt.ylabel('Latent dim 2')
plt.title('Autoencoder (2D) - digits')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ==========================
# 7. Guía breve para la conclusión
# ==========================

# Comentario (no ejecutable):
# - Observa los tres tipos de gráficos: PCA, t-SNE (perps 30 y 50) y autoencoder.
# - Evalúa:
#   * ¿En cuál se ven clusters de dígitos más separados?
#   * ¿Cuál parece más sensible al cambio de parámetros (perplejidad, arquitectura, etc.)?
# - Escribe una conclusión justificando cuál técnica prefieres para este dataset y por qué.

