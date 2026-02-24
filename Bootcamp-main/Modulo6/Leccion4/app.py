# app.py - Auto MPG regresión múltiple en Streamlit (métrico)

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# -------------------------
# 1. Cargar y preparar datos
# -------------------------

@st.cache_data
def cargar_datos():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"

    column_names = [
        "mpg", "cylinders", "displacement", "horsepower",
        "weight", "acceleration", "model_year", "origin", "car_name",
    ]

    auto = pd.read_csv(
        url,
        names=column_names,
        na_values="?",
        comment="\t",
        sep=" ",
        skipinitialspace=True,
    )

    # Nos quedamos con columnas numéricas y eliminamos NaN
    auto = auto[
        ["mpg", "cylinders", "displacement", "horsepower",
         "weight", "acceleration", "model_year"]
    ].dropna()

    # Conversiones a métricas
    auto["weight_kg"] = auto["weight"] * 0.45359237
    auto["consumo_l_100km"] = 235.214583 / auto["mpg"]

    return auto


auto = cargar_datos()

# -------------------------
# 2. Entrenar modelo múltiple
# -------------------------

X = auto[["weight_kg", "horsepower", "cylinders"]]
y = auto["consumo_l_100km"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

lin_reg_multi = LinearRegression()
lin_reg_multi.fit(X_train, y_train)

y_pred = lin_reg_multi.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# -------------------------
# 3. Interfaz Streamlit
# -------------------------

st.title("Auto MPG - Regresión múltiple (Streamlit)")
st.write(
    "Modelo de regresión lineal que estima el consumo en **L/100 km** "
    "a partir del **peso (kg)**, **caballos de fuerza (hp)** y **cilindros**."
)

st.markdown(f"**R² del modelo:** `{r2:.3f}` &nbsp;&nbsp; **MAE:** `{mae:.2f} L/100 km`")

st.sidebar.header("Parámetros del vehículo")

peso_kg = st.sidebar.slider(
    "Peso (kg)",
    float(auto["weight_kg"].min()),
    float(auto["weight_kg"].max()),
    float(auto["weight_kg"].median()),
    step=50.0,
)

horsepower = st.sidebar.slider(
    "Caballos de fuerza (hp)",
    float(auto["horsepower"].min()),
    float(auto["horsepower"].max()),
    float(auto["horsepower"].median()),
    step=5.0,
)

cylinders = st.sidebar.selectbox(
    "Cilindros",
    sorted(auto["cylinders"].unique()),
)

# DataFrame con los mismos nombres de columnas que el entrenamiento
X_nuevo = pd.DataFrame(
    [[peso_kg, horsepower, cylinders]],
    columns=["weight_kg", "horsepower", "cylinders"],
)

consumo_pred = float(lin_reg_multi.predict(X_nuevo)[0])

st.subheader("Predicción del modelo")
st.metric("Consumo estimado (L/100 km)", round(consumo_pred, 2))

st.write(
    f"Para un auto de **{peso_kg:.0f} kg**, **{horsepower:.0f} hp** "
    f"y **{cylinders} cilindros**, el modelo estima "
    f"**{consumo_pred:.2f} L/100 km**."
)

st.subheader("Muestra de datos reales")

muestra = auto.sample(200, random_state=42)
st.scatter_chart(
    data=muestra,
    x="weight_kg",
    y="consumo_l_100km",
)
