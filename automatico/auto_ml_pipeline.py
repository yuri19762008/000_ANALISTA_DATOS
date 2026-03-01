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


# ======================
# Funciones auxiliares
# ======================

def limpiar_y_normalizar(df):
    # 1) Eliminar columnas/filas vacías y duplicados
    df = df.dropna(axis=1, how="all")
    df = df.dropna(axis=0, how="all")
    df = df.drop_duplicates()

    # 2) Detectar columnas numéricas y categóricas (object/string/category)
    cols_num = df.select_dtypes(include=[np.number]).columns
    cols_cat = df.select_dtypes(include=["object", "string", "category"]).columns

    # 3) Imputar numéricas y escalarlas
    imputer_num = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    if len(cols_num) > 0:
        df_num = pd.DataFrame(
            imputer_num.fit_transform(df[cols_num]),
            columns=cols_num,
            index=df.index
        )

        df_num_scaled = pd.DataFrame(
            scaler.fit_transform(df_num),
            columns=cols_num,
            index=df.index
        )
    else:
        df_num_scaled = pd.DataFrame(index=df.index)

    # 4) Codificar categóricas (one-hot encoding)
    if len(cols_cat) > 0:
        # get_dummies -> columnas 0/1 por categoría. [web:69][web:72][web:75]
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


def detectar_problema(y):
    if pd.api.types.is_numeric_dtype(y):
        n_unicos = y.nunique()
        if n_unicos <= 15:
            return "clasificacion"
        else:
            return "regresion"
    else:
        return "clasificacion"


def evaluar_modelos_clasificacion(X, y):
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


def evaluar_modelos_regresion(X, y):
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


def graficos_exploratorios(df, nombre_target=None):
    cols_num = df.select_dtypes(include=[np.number]).columns

    if len(cols_num) == 0:
        st.info("No hay columnas numéricas para graficar.")
        return

    # Histogramas
    st.subheader("Histogramas variables numéricas")
    fig, ax = plt.subplots(figsize=(12, 8))
    df[cols_num].hist(ax=ax, bins=20)
    plt.tight_layout()
    st.pyplot(fig)

    # Boxplots
    st.subheader("Boxplots variables numéricas")
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    df[cols_num].plot(kind="box", ax=ax2)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig2)

    # Pairplot
    if len(cols_num) <= 6:
        st.subheader("Pairplot variables numéricas")
        cols_para_pair = list(cols_num)
        if nombre_target is not None and nombre_target in df.columns and nombre_target not in cols_para_pair:
            cols_para_pair.append(nombre_target)
        pairplot_fig = sns.pairplot(df[cols_para_pair])
        st.pyplot(pairplot_fig.fig)


def graficos_modelo_clasificacion(modelo, X_test, y_test):
    y_pred = modelo.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    st.subheader("Matriz de confusión (mejor modelo)")
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues", ax=ax)
    st.pyplot(fig)


def graficos_modelo_regresion(modelo, X_test, y_test):
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
# App Streamlit
# ======================

st.title("Auto-pipeline: Limpieza, Codificación, Modelos y Gráficos")

st.markdown(
    "1. Sube un archivo CSV.\n"
    "2. Elige la columna objetivo (target).\n"
    "3. El sistema limpia, codifica categóricas, normaliza, compara modelos y genera gráficos automáticamente."
)

uploaded_file = st.file_uploader("Sube tu archivo CSV", type="csv")

if uploaded_file is not None:
    df_raw = pd.read_csv(uploaded_file)
    st.subheader("Vista previa de datos originales")
    st.dataframe(df_raw.head())

    columna_target = st.selectbox(
        "Selecciona la columna target",
        options=df_raw.columns
    )

    if st.button("Ejecutar análisis automático"):
        with st.spinner("Procesando..."):
            # Limpieza + codificación + normalización
            df_limpio = limpiar_y_normalizar(df_raw)

            st.subheader("Datos limpios/codificados/normalizados (primeras filas)")
            st.dataframe(df_limpio.head())

            # Gráficos exploratorios (sobre df_limpio)
            graficos_exploratorios(df_limpio, nombre_target=None)

            # Separar X, y usando la columna target original
            y = df_raw[columna_target]
            X = df_limpio.copy()

            # Detectar tipo problema
            tipo = detectar_problema(y)
            st.write(f"Tipo de problema detectado: **{tipo}**")

            # Split train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Evaluar modelos
            if tipo == "clasificacion":
                resultados, modelos = evaluar_modelos_clasificacion(X_train, y_train)
                metrica = "accuracy"
            else:
                resultados, modelos = evaluar_modelos_regresion(X_train, y_train)
                metrica = "R2"

            # Mostrar resultados en tabla
            st.subheader("Resultados de modelos")
            df_res = pd.DataFrame(
                [{"Modelo": k, "Score": v} for k, v in resultados.items()]
            ).sort_values("Score", ascending=False)
            st.dataframe(df_res)

            # Mejor modelo
            mejor_nombre = max(resultados, key=resultados.get)
            mejor_score = resultados[mejor_nombre]
            mejor_modelo = modelos[mejor_nombre]
            mejor_modelo.fit(X_train, y_train)

            st.success(f"Mejor modelo: {mejor_nombre} con {metrica} = {mejor_score:.4f}")

            # Gráficos del mejor modelo
            if tipo == "clasificacion":
                graficos_modelo_clasificacion(mejor_modelo, X_test, y_test)
            else:
                graficos_modelo_regresion(mejor_modelo, X_test, y_test)

            # Descargar datos limpios
            csv_limpio = df_limpio.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Descargar CSV limpio/codificado/normalizado",
                data=csv_limpio,
                file_name="datos_limpios_codificados_normalizados.csv",
                mime="text/csv",
            )
else:
    st.info("Sube un CSV para comenzar.")
