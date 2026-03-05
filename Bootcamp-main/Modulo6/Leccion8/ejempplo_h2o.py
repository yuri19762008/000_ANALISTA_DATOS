# ejemplo_h2o_ucimlrepo.py

from ucimlrepo import fetch_ucirepo
import pandas as pd

import h2o
from h2o.automl import H2OAutoML

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


def main():
    # 1) Dataset UCI id=15 (Breast Cancer Wisconsin Original)
    data = fetch_ucirepo(id=15)
    X = data.data.features.copy()
    y = data.data.targets.copy()

    # 2) Renombrar columnas a versión "clínica"
    rename_map = {
        "Clumpthickness": "Clump_thickness",
        "Uniformityofcellsize": "Uniformity_of_cell_size",
        "Uniformityofcellshape": "Uniformity_of_cell_shape",
        "Marginaladhesion": "Marginal_adhesion",
        "Singleepithelialcellsize": "Single_epithelial_cell_size",
        "Barenuclei": "Bare_nuclei",
        "Blandchromatin": "Bland_chromatin",
        "Normalnucleoli": "Normal_nucleoli",
        "Mitoses": "Mitoses",
    }
    X = X.rename(columns=rename_map)

    target_col = y.columns[0]

    # 3) Split train/test para evaluación tipo sklearn
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y[target_col],
        test_size=0.2,
        random_state=42,
        stratify=y[target_col],
    )

    # 4) Iniciar H2O y preparar frames
    h2o.init()

    # H2OFrame de train (unimos X_train + y_train)
    df_train = pd.concat([X_train, y_train], axis=1)
    hf_train = h2o.H2OFrame(df_train)

    hf_train[target_col] = hf_train[target_col].asfactor()

    x = [c for c in hf_train.columns if c != target_col]
    y_h2o = target_col

    # 5) AutoML
    aml = H2OAutoML(
        max_models=20,
        max_runtime_secs=120,
        seed=123,
    )
    aml.train(x=x, y=y_h2o, training_frame=hf_train)

    print("=== Leaderboard (Top 10) ===")
    print(aml.leaderboard.head(rows=10))

    leader = aml.leader
    print("\n=== Modelo líder ===")
    print(leader)

    # 6) Evaluación en test (scikit-learn)
    hf_test = h2o.H2OFrame(X_test)
    pred_test = leader.predict(hf_test).as_data_frame()
    y_pred = pred_test["predict"]

    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy en test: {acc:.4f}")
    print("\nClassification report:\n")
    print(classification_report(y_test, y_pred))

    # 7) Apagar H2O
    h2o.shutdown(prompt=False)


if __name__ == "__main__":
    main()
