"""demo_clustering_asociacion.py

Live coding: clustering avanzado y reglas de asociación en retail
Dataset: Online Retail / Online Retail II (UCI / Kaggle).

Pasos:
1) Carga y preparación de datos
2) Aplicación de DBSCAN
3) Visualización 2D con t-SNE
4) Agrupamiento jerárquico y dendrogramas
5) Aplicación de Apriori (mlxtend)
6) Filtrado de reglas significativas
7) Visualización de soporte/confianza/lift
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import dendrogram, linkage
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules




# ==============================
# 1) Carga y preparación de datos
# ==============================

def cargar_y_preparar_datos(file_path:  str, country: str = "United Kingdom"):
    """Carga el CSV de retail, limpia datos y construye features por cliente.

    Parameters
    ----------
    file_path : str
        Ruta al archivo CSV de Online Retail / Online Retail II.
    country : str
        País a filtrar para el análisis.

    Returns
    -------
    df : DataFrame
        Datos de detalle (línea de factura) limpios.
    customer_df : DataFrame
        Datos agregados por cliente con variables numéricas.
    X : ndarray
        Matriz estandarizada para clustering.
    """
    df = pd.read_csv(file_path, encoding="ISO-8859-1")
    print("Shape original:", df.shape)

    df = df.dropna(subset=["Description", "Customer ID"])
    df = df[~df["Invoice"].astype(str).str.startswith("C")]
    print("Shape tras limpiar nulos y devoluciones:", df.shape)

    df = df[df["Country"] == country]
    print(f"Shape tras filtrar país ({country}):", df.shape)

    df["TotalPrice"] = df["Quantity"] * df["Price"]

    customer_df = (
        df.groupby("Customer ID")
          .agg(
              NumInvoices=("Invoice", "nunique"),
              NumItems=("Quantity", "sum"),
              TotalSpent=("TotalPrice", "sum")
          )
          .reset_index()
    )
    print("Clientes únicos:", customer_df.shape[0])

    print(customer_df.dtypes)

    for col in ["NumInvoices", "NumItems", "TotalSpent"]:
        customer_df[col] = (
        customer_df[col]
        .astype(str)
        .str.replace(",", ".", regex=False)  # si usara coma decimal
        .astype(float)
    )


    
    features = ["NumInvoices", "NumItems", "TotalSpent"]
    scaler = StandardScaler()
    X = scaler.fit_transform(customer_df[features])
    print("Shape de X para clustering:", X.shape)

    return df, customer_df, X


# ======================
# 2) Aplicación de DBSCAN
# ======================

def aplicar_dbscan(X, customer_df, eps: float = 0.8, min_samples: int = 10):
    """Aplica DBSCAN y devuelve labels y customer_df actualizado."""
    neighbors = NearestNeighbors(n_neighbors=5)
    neighbors_fit = neighbors.fit(X)
    distances, _ = neighbors_fit.kneighbors(X)
    distances = np.sort(distances[:, -1])

    plt.figure(figsize=(6, 4))
    plt.plot(distances)
    plt.ylabel("Distancia al 5º vecino")
    plt.xlabel("Puntos ordenados")
    plt.title("Curva k-dist para seleccionar eps")
    plt.grid(True)
    plt.show()

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X)

    customer_df = customer_df.copy()
    customer_df["DBSCAN_cluster"] = labels

    unique, counts = np.unique(labels, return_counts=True)
    cluster_info = dict(zip(unique, counts))
    print("Tamaño de cada cluster (incluyendo -1 como ruido):")
    for k, v in cluster_info.items():
        print(f"Cluster {k}: {v} clientes")

    return labels, customer_df


# ==================================
# 3) Visualización 2D con t-SNE
# ==================================

def visualizar_tsne(X, labels, title: str = "Clientes con t-SNE y DBSCAN"):
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X)

    plt.figure(figsize=(7, 6))
    scatter = plt.scatter(
        X_tsne[:, 0],
        X_tsne[:, 1],
        c=labels,
        cmap="tab10",
        s=15,
        alpha=0.8,
    )
    plt.colorbar(scatter, label="Cluster")
    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.grid(True)
    plt.show()


# ===============================================
# 4) Clustering jerárquico y dendrogramas
# ===============================================

def clustering_jerarquico(X, customer_df, n_clusters: int = 4, sample_size: int = 300):
    """Dibuja un dendrograma y asigna clusters jerárquicos."""
    sample_size = min(sample_size, X.shape[0])
    np.random.seed(42)
    sample_idx = np.random.choice(len(X), size=sample_size, replace=False)
    X_sample = X[sample_idx]

    Z = linkage(X_sample, method="ward")

    plt.figure(figsize=(10, 4))
    dendrogram(
        Z,
        truncate_mode="lastp",
        p=20,
        leaf_rotation=45.0,
        leaf_font_size=8.0,
    )
    plt.title("Dendrograma truncado (ward)")
    plt.xlabel("Clusters")
    plt.ylabel("Distancia")
    plt.show()

    agg = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    labels_hier = agg.fit_predict(X)

    customer_df = customer_df.copy()
    customer_df["HIER_cluster"] = labels_hier

    unique, counts = np.unique(labels_hier, return_counts=True)
    cluster_info = dict(zip(unique, counts))
    print("Tamaño de cada cluster jerárquico:")
    for k, v in cluster_info.items():
        print(f"Cluster {k}: {v} clientes")

    return labels_hier, customer_df


# ======================================
# 5) Apriori (mlxtend) e itemsets frecuentes
# ======================================

def preparar_transacciones(df):
    """Convierte el detalle de facturas en lista de transacciones."""
    basket_series = df.groupby("Invoice")["Description"].apply(list)
    transactions = basket_series.tolist()
    print("Número de transacciones:", len(transactions))
    return transactions


def aplicar_apriori(transactions, min_support: float = 0.02):
    te = TransactionEncoder()
    te_array = te.fit(transactions).transform(transactions)

    basket_df = pd.DataFrame(te_array, columns=te.columns_)
    print("Shape de la matriz one-hot:", basket_df.shape)

    frequent_itemsets = apriori(
        basket_df,
        min_support=min_support,
        use_colnames=True,
    )
    frequent_itemsets = frequent_itemsets.sort_values("support", ascending=False)
    print("Número de itemsets frecuentes:", frequent_itemsets.shape[0])

    return basket_df, frequent_itemsets


# ======================================
# 6) Reglas de asociación y filtrado
# ======================================

def generar_y_filtrar_reglas(frequent_itemsets,
                             min_confidence: float = 0.3,
                             support_threshold: float = 0.02,
                             confidence_threshold: float = 0.4,
                             lift_threshold: float = 1.2):
    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=min_confidence,
    )
    print("Número de reglas generadas (sin filtro extra):", rules.shape[0])

    rules_filtered = rules[
        (rules["support"] >= support_threshold)
        & (rules["confidence"] >= confidence_threshold)
        & (rules["lift"] >= lift_threshold)
    ].sort_values("lift", ascending=False)

    print("Reglas tras el filtrado:", rules_filtered.shape[0])
    return rules, rules_filtered


# ======================================
# 7) Visualización de reglas
# ======================================

def visualizar_reglas(rules_filtered):
    if rules_filtered.empty:
        print("No hay reglas que cumplan los umbrales establecidos.")
        return

    plt.figure(figsize=(7, 6))
    scatter = plt.scatter(
        rules_filtered["support"],
        rules_filtered["confidence"],
        c=rules_filtered["lift"],
        cmap="viridis",
        s=40,
        alpha=0.7,
    )
    plt.colorbar(scatter, label="Lift")
    plt.xlabel("Soporte")
    plt.ylabel("Confianza")
    plt.title("Reglas de asociación: soporte vs confianza (color = lift)")
    plt.grid(True)
    plt.show()


# ======================================
# Ejecución de ejemplo
# ======================================

if __name__ == "__main__":
    # 1) Carga y preparación
    file_path = r"D:\000_ANALISTA DATOS , TALENTO DIGITAL\Bootcamp-main\Modulo7\Leccion1\online_retail_II.csv"
    # ajusta la ruta a tu CSV
    df, customer_df, X = cargar_y_preparar_datos(file_path)

    # 2) DBSCAN
    labels_dbscan, customer_df = aplicar_dbscan(X, customer_df, eps=0.8, min_samples=10)

    # 3) t-SNE
    visualizar_tsne(X, labels_dbscan, title="Clientes con t-SNE coloreados por DBSCAN")

    # 4) Jerárquico
    labels_hier, customer_df = clustering_jerarquico(X, customer_df, n_clusters=4, sample_size=300)

    # 5) Apriori
    transactions = preparar_transacciones(df)
    basket_df, frequent_itemsets = aplicar_apriori(transactions, min_support=0.02)

    # 6) Reglas
    rules, rules_filtered = generar_y_filtrar_reglas(
        frequent_itemsets,
        min_confidence=0.3,
        support_threshold=0.02,
        confidence_threshold=0.4,
        lift_threshold=1.2,
    )

    # 7) Visualización de reglas
    visualizar_reglas(rules_filtered)
