import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split

# Modelos de clasificación
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Modelos de regresión
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

# Métricas
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, r2_score

sns.set()


# ======================
# Limpieza + codificación + normalización
# ======================
def limpiar_y_normalizar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Paso a paso:
    1) Eliminar columnas/filas completamente vacías y filas duplicadas.
    2) Separar columnas numéricas y categóricas.
    3) Para las numéricas: imputar valores faltantes con la mediana y luego escalar.
    4) Para las categóricas: aplicar one-hot encoding (variables 0/1).
    5) Unir todo en un solo DataFrame numérico, listo para modelos.
    """

    # 1) Eliminar columnas/filas vacías y duplicados
    df = df.dropna(axis=1, how="all")   # elimina columnas donde todos los valores son NaN
    df = df.dropna(axis=0, how="all")   # elimina filas donde todos los valores son NaN
    df = df.drop_duplicates()           # elimina filas repetidas

    # 2) Detectar columnas numéricas y categóricas
    cols_num = df.select_dtypes(include=[np.number]).columns
    cols_cat = df.select_dtypes(include=["object", "string", "category"]).columns

    # 3) Imputar numéricas y escalarlas
    imputer_num = SimpleImputer(strategy="median")  # mediana es robusta a outliers
    scaler = StandardScaler()                       # deja media=0 y desviación estándar=1

    if len(cols_num) > 0:
        # 3.1) Imputar valores faltantes en columnas numéricas
        df_num = pd.DataFrame(
            imputer_num.fit_transform(df[cols_num]),
            columns=cols_num,
            index=df.index
        )

        # 3.2) Escalar columnas numéricas
        df_num_scaled = pd.DataFrame(
            scaler.fit_transform(df_num),
            columns=cols_num,
            index=df.index
        )
    else:
        df_num_scaled = pd.DataFrame(index=df.index)

    # 4) Codificar categóricas (one-hot encoding)
    # Cada categoría se convierte en una columna 0/1.
    if len(cols_cat) > 0:
        df_cat_encoded = pd.get_dummies(
            df[cols_cat],
            drop_first=False,
            dtype=int
        )
    else:
        df_cat_encoded = pd.DataFrame(index=df.index)

    # 5) Unir numéricas escaladas + categóricas codificadas
    df_final = pd.concat([df_num_scaled, df_cat_encoded], axis=1)

    return df_final


# ======================
# Tipo de problema y evaluación de modelos
# ======================
def detectar_problema(y: pd.Series) -> str:
    """
    Heurística para decidir si el problema es:
    - Clasificación: pocas clases (<= 15) o variable no numérica.
    - Regresión: variable numérica continua con muchas clases.
    """
    if pd.api.types.is_numeric_dtype(y):
        n_unicos = y.nunique()
        if n_unicos <= 15:
            return "clasificacion"
        else:
            return "regresion"
    else:
        return "clasificacion"


def evaluar_modelos_clasificacion(X: pd.DataFrame, y: pd.Series):
    """
    Clasificación:
    1) Define un diccionario de modelos.
    2) Para cada modelo, realiza validación cruzada (cv=5) con métrica accuracy.
    3) Devuelve un diccionario con el promedio de accuracy por modelo y el diccionario de modelos.
    """
    modelos = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForestClassifier": RandomForestClassifier(random_state=42),
        "SVC": SVC(probability=True)
    }
    resultados = {}

    for nombre, modelo in modelos.items():
        scores = cross_val_score(modelo, X, y, cv=5, scoring="accuracy")
        resultados[nombre] = scores.mean()

    return resultados, modelos


def evaluar_modelos_regresion(X: pd.DataFrame, y: pd.Series):
    """
    Regresión:
    1) Define un diccionario de modelos.
    2) Para cada modelo, realiza validación cruzada (cv=5) con métrica R2.
    3) Devuelve un diccionario con el promedio de R2 por modelo y el diccionario de modelos.
    """
    modelos = {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(random_state=42),
        "SVR": SVR()
    }
    resultados = {}

    for nombre, modelo in modelos.items():
        scores = cross_val_score(modelo, X, y, cv=5, scoring="r2")
        resultados[nombre] = scores.mean()

    return resultados, modelos


# ======================
# Gráficos exploratorios
# ======================
def graficos_exploratorios(df_limpio: pd.DataFrame, df_raw: pd.DataFrame):
    """
    Genera:
    - Histogramas y boxplots de variables numéricas (df_limpio).
    - Pairplot de numéricas si no hay demasiadas columnas.
    - Gráfico de barras de una variable categórica de df_raw.
    """
    cols_num = df_limpio.select_dtypes(include=[np.number]).columns

    if len(cols_num) == 0:
        st.info("No hay columnas numéricas para graficar.")
        return

    # Histogramas
    st.subheader("Histogramas variables numéricas")
    fig, ax = plt.subplots(figsize=(12, 8))
    df_limpio[cols_num].hist(ax=ax, bins=20)
    plt.tight_layout()
    st.pyplot(fig)

    # Boxplots
    st.subheader("Boxplots variables numéricas")
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    df_limpio[cols_num].plot(kind="box", ax=ax2)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig2)

    # Pairplot (si pocas columnas)
    if len(cols_num) <= 6:
        st.subheader("Pairplot variables numéricas")
        pairplot_fig = sns.pairplot(df_limpio[cols_num])
        st.pyplot(pairplot_fig.fig)

    # Frecuencias categóricas desde datos crudos
    cols_cat_raw = df_raw.select_dtypes(include=["object", "string", "category"]).columns
    if len(cols_cat_raw) > 0:
        st.subheader("Frecuencias de una variable categórica (datos originales)")
        col_cat_sel = st.selectbox("Elige una variable categórica para graficar", cols_cat_raw)
        fig3, ax3 = plt.subplots()
        df_raw[col_cat_sel].value_counts().plot(kind="bar", rot=45, figsize=(6, 4), ax=ax3)
        plt.tight_layout()
        st.pyplot(fig3)


# ======================
# Gráficos de desempeño de modelos
# ======================
def graficos_modelo_clasificacion(modelo, X_test: pd.DataFrame, y_test: pd.Series):
    """
    Genera la matriz de confusión del mejor modelo de clasificación.
    """
    y_pred = modelo.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    st.subheader("Matriz de confusión (mejor modelo)")
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues", ax=ax)
    st.pyplot(fig)


def graficos_modelo_regresion(modelo, X_test: pd.DataFrame, y_test: pd.Series):
    """
    Genera el gráfico de valores reales vs predichos para el mejor modelo de regresión.
    """
    y_pred = modelo.predict(X_test)

    st.subheader("Predicho vs Real (mejor modelo)")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, y_pred, alpha=0.7)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--")
    ax.set_xlabel("Valores reales")
    ax.set_ylabel("Valores predichos")
    ax.set_title(f"R2 = {r2_score(y_test, y_pred):.3f}")
    plt.tight_layout()
    st.pyplot(fig)


# ======================
# App Streamlit (flujo completo)
# ======================
st.title("Auto-pipeline CSV: Limpieza, Codificación, Modelos y Gráficos")

st.markdown(
    """
    **Flujo de la app:**

    1. Sube un archivo CSV.
    2. La app muestra una vista previa y estadísticas descriptivas.
    3. Se limpian los datos, se codifican variables categóricas y se normalizan numéricas.
    4. Se generan gráficos exploratorios.
    5. Eliges la columna objetivo (target).
    6. Se detecta si es clasificación o regresión.
    7. Se prueban varios modelos, se comparan y se elige el mejor.
    8. Se muestran métricas y gráficos del mejor modelo.
    9. Puedes descargar el CSV limpio/codificado/normalizado.
    """
)

uploaded_file = st.file_uploader("Sube tu archivo CSV", type="csv")

if uploaded_file is not None:
    # 1) Cargar datos originales
    df_raw = pd.read_csv(uploaded_file)
    st.subheader("Vista previa de datos originales")
    st.dataframe(df_raw.head())

    st.subheader("Estadísticas descriptivas")
    st.write(df_raw.describe(include="all"))

    # 2) Elegir target
    columna_target = st.selectbox(
        "Selecciona la columna target (variable objetivo)",
        options=df_raw.columns
    )

    if st.button("Ejecutar análisis automático"):
        with st.spinner("Procesando..."):
            # 3) Limpieza + codificación + normalización
            df_limpio = limpiar_y_normalizar(df_raw)

            st.subheader("Datos limpios/codificados/normalizados (primeras filas)")
            st.dataframe(df_limpio.head())

            st.subheader("Columnas después de la codificación")
            st.write(df_limpio.columns.tolist())

            # 4) Gráficos exploratorios
            graficos_exploratorios(df_limpio, df_raw)

            # 5) Separar X, y (target desde raw, features desde df_limpio)
            y = df_raw[columna_target]
            X = df_limpio.copy()

            # 6) Detectar tipo problema
            tipo = detectar_problema(y)
            st.write(f"Tipo de problema detectado: **{tipo}**")

            # 7) Split train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # 8) Evaluar modelos
            if tipo == "clasificacion":
                resultados, modelos = evaluar_modelos_clasificacion(X_train, y_train)
                metrica = "accuracy"
            else:
                resultados, modelos = evaluar_modelos_regresion(X_train, y_train)
                metrica = "R2"

            # 9) Mostrar resultados
            st.subheader("Resultados de modelos")
            df_res = pd.DataFrame(
                [{"Modelo": k, "Score": v} for k, v in resultados.items()]
            ).sort_values("Score", ascending=False)
            st.dataframe(df_res)

            # 10) Mejor modelo
            mejor_nombre = max(resultados, key=resultados.get)
            mejor_score = resultados[mejor_nombre]
            mejor_modelo = modelos[mejor_nombre]
            mejor_modelo.fit(X_train, y_train)

            st.success(f"Mejor modelo: {mejor_nombre} con {metrica} = {mejor_score:.4f}")
            st.markdown(
                """
                - **Accuracy** (clasificación): proporción de aciertos sobre el total.
                - **R2** (regresión): proporción de variabilidad explicada por el modelo.
                """
            )

            # 11) Gráficos del mejor modelo
            if tipo == "clasificacion":
                graficos_modelo_clasificacion(mejor_modelo, X_test, y_test)
            else:
                graficos_modelo_regresion(mejor_modelo, X_test, y_test)

            # 12) Descargar datos limpios
            csv_limpio = df_limpio.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Descargar CSV limpio/codificado/normalizado",
                data=csv_limpio,
                file_name="datos_limpios_codificados_normalizados.csv",
                mime="text/csv",
            )
else:
    st.info("Sube un CSV para comenzar.")
