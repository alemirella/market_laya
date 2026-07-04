# -*- coding: utf-8 -*-
"""
=============================================================
  MARKET LAYA — Evaluación de Modelos (separada del entrenamiento)
=============================================================
NUEVO MÓDULO (mejora #5). En la versión anterior del proyecto, toda
la evaluación (accuracy, reporte de clasificación, matriz de
confusión) vivía mezclada dentro de modelo_ia_market_laya.py, en el
mismo script que entrena el modelo de producción. Eso dificultaba:
  - reutilizar la evaluación con otros modelos/variantes,
  - correr solo la evaluación sin reentrenar todo,
  - documentar la metodología de evaluación de forma independiente
    para el capítulo de resultados de la tesis/informe.

Este script SOLO evalúa (no decide el modelo de producción, ese rol
lo mantiene modelos/modelo_riesgo.py). Aquí se agregan además dos
análisis que antes no existían:
  1) VALIDACIÓN CRUZADA (K-Fold estratificado) sobre el set de
     entrenamiento, para tener una estimación de la varianza del
     desempeño del modelo, no solo un único punto (el split temporal).
  2) COMPARACIÓN TEMPORAL: se repite la evaluación en varios cortes
     de tiempo sucesivos (ventanas expansivas) para observar si el
     desempeño del modelo se degrada o mejora con el tiempo/con más
     datos disponibles — importante en un problema con fuerte
     desbalance de clases y relativamente pocos lotes "vencibles".

Ejecutar desde la raíz del proyecto:
    python evaluacion/evaluacion_modelos.py
Genera:
    resultados/matriz_confusion.png
    resultados/comparacion_temporal.png
    (además imprime en consola el resumen de validación cruzada)
=============================================================
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from modelos.modelo_riesgo import (
    cargar_datos, construir_dataset, clasificar_riesgo,
    FEATURES, TARGET, CLASES_ORDEN,
)
from sklearn.preprocessing import LabelEncoder

RESULTADOS_DIR = os.path.join(BASE_DIR, "resultados")


# ─────────────────────────────────────────────────────────────
# 1. PREPARAR DATASET ETIQUETADO (misma lógica que modelo_riesgo.py)
# ─────────────────────────────────────────────────────────────
def _preparar_dataset_train():
    inventario, ventas, rotacion = cargar_datos()
    dataset = construir_dataset(inventario, ventas, rotacion)
    dataset["nivel_riesgo"] = dataset.apply(clasificar_riesgo, axis=1)
    dataset_train = dataset[dataset["nivel_riesgo"] != "VENCIDO"].copy()

    le_cat = LabelEncoder()
    dataset_train["categoria_enc"] = le_cat.fit_transform(dataset_train["categoria"])
    dataset_train = dataset_train.sort_values("fecha_ingreso").reset_index(drop=True)
    return dataset_train


# ─────────────────────────────────────────────────────────────
# 2. VALIDACIÓN CRUZADA ESTRATIFICADA (K-Fold)
# ─────────────────────────────────────────────────────────────
def validacion_cruzada(dataset_train, k=5):
    X = dataset_train[FEATURES]
    y = dataset_train[TARGET]

    # k no puede superar el número de muestras de la clase más pequeña
    min_clase = y.value_counts().min()
    k_efectivo = max(2, min(k, int(min_clase)))
    if k_efectivo < k:
        print(f"  Aviso: se reduce k={k} a k={k_efectivo} porque la clase minoritaria "
              f"solo tiene {min_clase} muestra(s).")

    modelo = RandomForestClassifier(
        n_estimators=100, max_depth=10, min_samples_leaf=2,
        random_state=42, class_weight="balanced",
    )
    skf = StratifiedKFold(n_splits=k_efectivo, shuffle=True, random_state=42)

    scores_f1_macro = cross_val_score(modelo, X, y, cv=skf, scoring="f1_macro")
    scores_acc = cross_val_score(modelo, X, y, cv=skf, scoring="accuracy")

    resumen = {
        "k": k_efectivo,
        "f1_macro_media": round(float(scores_f1_macro.mean()) * 100, 2),
        "f1_macro_std": round(float(scores_f1_macro.std()) * 100, 2),
        "accuracy_media": round(float(scores_acc.mean()) * 100, 2),
        "accuracy_std": round(float(scores_acc.std()) * 100, 2),
        "folds_f1_macro": [round(float(s) * 100, 2) for s in scores_f1_macro],
        "folds_accuracy": [round(float(s) * 100, 2) for s in scores_acc],
    }
    return resumen


# ─────────────────────────────────────────────────────────────
# 3. MATRIZ DE CONFUSIÓN VISUAL (sobre un split temporal fijo)
# ─────────────────────────────────────────────────────────────
def _split_temporal_estratificado(dataset_train, test_size=0.2):
    train_frames, test_frames = [], []
    for _clase, grupo in dataset_train.groupby("nivel_riesgo"):
        grupo = grupo.sort_values("fecha_ingreso")
        corte = max(1, int(len(grupo) * (1 - test_size)))
        corte = min(corte, len(grupo) - 1) if len(grupo) > 1 else corte
        train_frames.append(grupo.iloc[:corte])
        test_frames.append(grupo.iloc[corte:])
    return pd.concat(train_frames).sort_index(), pd.concat(test_frames).sort_index()


def matriz_confusion_visual(dataset_train, ruta_salida):
    train_df, test_df = _split_temporal_estratificado(dataset_train)
    modelo = RandomForestClassifier(
        n_estimators=100, max_depth=10, min_samples_leaf=2,
        random_state=42, class_weight="balanced",
    )
    modelo.fit(train_df[FEATURES], train_df[TARGET])
    y_pred = modelo.predict(test_df[FEATURES])

    clases_presentes = [c for c in CLASES_ORDEN if c in test_df[TARGET].unique()]
    cm = confusion_matrix(test_df[TARGET], y_pred, labels=clases_presentes)

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clases_presentes)
    disp.plot(ax=ax, cmap="Blues", colorbar=True, values_format="d")
    ax.set_title("Matriz de confusión — Modelo de riesgo (split temporal)")
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=150)
    plt.close(fig)
    return cm, clases_presentes


# ─────────────────────────────────────────────────────────────
# 4. COMPARACIÓN TEMPORAL (ventanas expansivas)
#    Reentrena y evalúa en varios cortes sucesivos de tiempo para
#    ver si el desempeño se mantiene estable a medida que avanza
#    el histórico de datos disponible.
# ─────────────────────────────────────────────────────────────
def comparacion_temporal(dataset_train, n_cortes=5, ruta_salida=None):
    dataset_train = dataset_train.sort_values("fecha_ingreso").reset_index(drop=True)
    fechas_unicas = sorted(dataset_train["fecha_ingreso"].unique())

    if len(fechas_unicas) < n_cortes + 1:
        n_cortes = max(2, len(fechas_unicas) - 1)

    puntos_corte = np.linspace(0.4, 0.9, n_cortes)  # % del histórico usado como frontera train/test
    resultados = []

    for pct in puntos_corte:
        idx_corte = int(len(dataset_train) * pct)
        if idx_corte < 5 or idx_corte >= len(dataset_train) - 2:
            continue
        train_df = dataset_train.iloc[:idx_corte]
        test_df = dataset_train.iloc[idx_corte: idx_corte + max(10, int(len(dataset_train) * 0.1))]

        if train_df[TARGET].nunique() < 2 or len(test_df) == 0:
            continue

        modelo = RandomForestClassifier(
            n_estimators=100, max_depth=10, min_samples_leaf=2,
            random_state=42, class_weight="balanced",
        )
        modelo.fit(train_df[FEATURES], train_df[TARGET])
        y_pred = modelo.predict(test_df[FEATURES])
        f1_macro = f1_score(test_df[TARGET], y_pred, average="macro", zero_division=0)

        resultados.append({
            "pct_historico_train": round(float(pct) * 100, 1),
            "n_train": len(train_df),
            "n_test": len(test_df),
            "f1_macro_pct": round(float(f1_macro) * 100, 2),
        })

    df_resultados = pd.DataFrame(resultados)

    if ruta_salida and len(df_resultados) > 0:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(df_resultados["pct_historico_train"], df_resultados["f1_macro_pct"], marker="o")
        ax.set_xlabel("% del histórico usado como entrenamiento")
        ax.set_ylabel("F1-macro (%)")
        ax.set_title("Comparación temporal — estabilidad del modelo de riesgo")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(ruta_salida, dpi=150)
        plt.close(fig)

    return df_resultados


# ─────────────────────────────────────────────────────────────
# 5. EJECUCIÓN COMO SCRIPT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  MARKET LAYA — Evaluación de modelos (independiente del entrenamiento)")
    print("=" * 60)

    os.makedirs(RESULTADOS_DIR, exist_ok=True)
    dataset_train = _preparar_dataset_train()

    print("\n[1] Validación cruzada estratificada (K-Fold)...")
    resumen_cv = validacion_cruzada(dataset_train, k=5)
    print(f"    k={resumen_cv['k']}")
    print(f"    F1-macro : media={resumen_cv['f1_macro_media']}%  std={resumen_cv['f1_macro_std']}%")
    print(f"    Accuracy : media={resumen_cv['accuracy_media']}%  std={resumen_cv['accuracy_std']}%")
    print(f"    F1-macro por fold : {resumen_cv['folds_f1_macro']}")

    print("\n[2] Matriz de confusión visual (split temporal)...")
    ruta_cm = os.path.join(RESULTADOS_DIR, "matriz_confusion.png")
    cm, clases = matriz_confusion_visual(dataset_train, ruta_cm)
    print(f"    Clases evaluadas: {clases}")
    print(f"    → {ruta_cm}")

    print("\n[3] Comparación temporal (ventanas expansivas)...")
    ruta_temporal = os.path.join(RESULTADOS_DIR, "comparacion_temporal.png")
    df_temporal = comparacion_temporal(dataset_train, n_cortes=5, ruta_salida=ruta_temporal)
    if len(df_temporal):
        print(df_temporal.to_string(index=False))
        print(f"    → {ruta_temporal}")
    else:
        print("    No hay suficientes datos para construir la comparación temporal.")

    print("\n" + "=" * 60)
    print("  Evaluación completa.")
    print("=" * 60)
