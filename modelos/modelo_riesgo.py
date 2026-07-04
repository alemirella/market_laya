# -*- coding: utf-8 -*-
"""
=============================================================
  MARKET LAYA - Modelo de IA: Clasificación de Riesgo de Vencimiento
  Asignatura : Inteligencia de Negocios
  Universidad: Universidad Continental
  Algoritmo  : Random Forest (Clasificación Supervisada)

  (antes: modelo_ia_market_laya.py)

  SIN CAMBIOS CONCEPTUALES respecto a la versión anterior:
  - El modelo sigue sin recibir 'dias_para_vencer' ni
    'stock_disponible' del lote actual como variables de entrada,
    porque esas dos variables son las que definen la etiqueta por
    regla de negocio (clasificar_riesgo). Dejarlas como input hacía
    que el modelo memorice la regla en vez de aprender a anticipar
    el riesgo con información disponible ANTES de conocer el lote.
  - Se sigue evaluando con una división TEMPORAL, estratificada por
    clase (entrena con lotes antiguos, prueba con los más recientes,
    respetando la proporción de cada nivel de riesgo).

  MEJORAS DE ESTA VERSIÓN:
  (#2) Balanceo de clases con SMOTE sobre el set de entrenamiento
       (además de class_weight="balanced"), porque el desbalance
       de clases (BAJO ≈ mayoría absoluta) hacía que el modelo casi
       no aprendiera patrones de las clases minoritarias (CRITICO/
       ALTO). Si SMOTE no puede aplicarse (muy pocas muestras en
       alguna clase para generar vecinos sintéticos, o la librería
       imbalanced-learn no está instalada), el script cae de forma
       automática y transparente a usar solo class_weight="balanced".
  (#5) Ya NO se reporta solo accuracy global: se guardan métricas
       por clase (precision/recall/F1), matriz de confusión y
       F1-macro/F1-ponderado en resultados/metricas_evaluacion.json,
       para poder sustentar el desempeño real del modelo por nivel
       de riesgo (crítico para la sección de resultados del informe).

  Este script se puede ejecutar de forma independiente:
      python modelos/modelo_riesgo.py
  (ejecutar desde la raíz del proyecto, market_laya/), y exporta:
      resultados/predicciones_riesgo_vencimiento.csv
      resultados/metricas_evaluacion.json
  También expone entrenar_modelo_riesgo(), que reutiliza app.py para
  no duplicar la lógica de entrenamiento entre el script standalone
  y el servidor Flask.
=============================================================
"""

import os
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
)

warnings.filterwarnings("ignore")

try:
    from imblearn.over_sampling import SMOTE
    _SMOTE_DISPONIBLE = True
except ImportError:
    _SMOTE_DISPONIBLE = False

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR_DEFAULT = os.path.join(BASE_DIR, "data")
RESULTADOS_DIR_DEFAULT = os.path.join(BASE_DIR, "resultados")

FEATURES = [
    "categoria_enc",          # Categoría del producto (codificada)
    "costo_unitario",         # Costo de compra
    "tasa_rotacion",          # Velocidad histórica de salida del producto
    "unidades_vendidas",      # Total histórico vendido del producto
    "ventas_30d",             # Ventas del producto en el último mes
    "precio_venta_promedio",  # Precio promedio de venta
    "stock_promedio",         # Stock promedio histórico del producto
]
# NOTA: 'dias_para_vencer' y 'stock_disponible' NO están aquí.
# Son las variables que definen la etiqueta (ver clasificar_riesgo),
# así que dejarlas como input hacía que el modelo memorice la regla
# en vez de aprender a anticipar el riesgo.
TARGET = "nivel_riesgo"
CLASES_ORDEN = ["CRITICO", "ALTO", "MODERADO", "BAJO"]


# ─────────────────────────────────────────────────────────────
# 1. CARGA Y FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────
def cargar_datos(data_dir=DATA_DIR_DEFAULT):
    inventario = pd.read_csv(os.path.join(data_dir, "inventario_limpio.csv"))
    ventas = pd.read_csv(os.path.join(data_dir, "ventas_limpio.csv"))
    rotacion = pd.read_csv(os.path.join(data_dir, "rotacion_productos.csv"))
    return inventario, ventas, rotacion


def construir_dataset(inventario, ventas, rotacion):
    """Arma el dataset con todas las variables (features + variables
    de etiquetado). Se reutiliza tanto para entrenar como para generar
    predicciones sobre el inventario actual."""
    inventario = inventario.copy()
    ventas = ventas.copy()

    inventario["fecha_ingreso"] = pd.to_datetime(inventario["fecha_ingreso"])
    inventario["fecha_vencimiento"] = pd.to_datetime(inventario["fecha_vencimiento"])
    inventario["dias_para_vencer"] = (
        inventario["fecha_vencimiento"] - inventario["fecha_ingreso"]
    ).dt.days

    rotacion_sel = rotacion[[
        "id_producto", "tasa_rotacion", "unidades_vendidas",
        "ingresos_totales", "stock_promedio",
    ]].copy()
    dataset = inventario.merge(rotacion_sel, on="id_producto", how="left")

    precio_promedio = (
        ventas.groupby("id_producto")["precio_venta"].mean().reset_index()
        .rename(columns={"precio_venta": "precio_venta_promedio"})
    )
    dataset = dataset.merge(precio_promedio, on="id_producto", how="left")

    ventas["fecha"] = pd.to_datetime(ventas["fecha"])
    fecha_ref = ventas["fecha"].max()
    ventas_recientes = ventas[ventas["fecha"] >= fecha_ref - pd.Timedelta(days=30)]
    ventas_30d = (
        ventas_recientes.groupby("id_producto")["cantidad_vendida"].sum().reset_index()
        .rename(columns={"cantidad_vendida": "ventas_30d"})
    )
    dataset = dataset.merge(ventas_30d, on="id_producto", how="left")
    dataset["ventas_30d"] = dataset["ventas_30d"].fillna(0)

    for col in ["tasa_rotacion", "unidades_vendidas", "ingresos_totales",
                "stock_promedio", "precio_venta_promedio"]:
        dataset[col] = dataset[col].fillna(dataset[col].median())

    return dataset


# ─────────────────────────────────────────────────────────────
# 2. ETIQUETADO (Variable Objetivo) — regla de negocio, sin cambios
# ─────────────────────────────────────────────────────────────
def clasificar_riesgo(row):
    dias = row["dias_para_vencer"]
    stock = row["stock_disponible"]
    rotacion_val = row["tasa_rotacion"] if row["tasa_rotacion"] > 0 else 1
    dias_agotamiento = stock / (rotacion_val / 365) if rotacion_val > 0 else 9999

    if dias <= 0:
        return "VENCIDO"
    elif dias <= 7 or (dias <= 20 and dias_agotamiento > dias):
        return "CRITICO"
    elif dias <= 15:
        return "ALTO"
    elif dias <= 30:
        return "MODERADO"
    else:
        return "BAJO"


# ─────────────────────────────────────────────────────────────
# 3. DIVISIÓN TRAIN/TEST TEMPORAL, ESTRATIFICADA POR CLASE
# ─────────────────────────────────────────────────────────────
def _split_temporal_estratificado(dataset_train, test_size=0.2):
    dataset_train = dataset_train.sort_values("fecha_ingreso").reset_index(drop=True)
    train_frames, test_frames = [], []
    for _clase, grupo in dataset_train.groupby("nivel_riesgo"):
        grupo = grupo.sort_values("fecha_ingreso")
        corte = max(1, int(len(grupo) * (1 - test_size)))
        corte = min(corte, len(grupo) - 1) if len(grupo) > 1 else corte
        train_frames.append(grupo.iloc[:corte])
        test_frames.append(grupo.iloc[corte:])
    train_df = pd.concat(train_frames).sort_index()
    test_df = pd.concat(test_frames).sort_index()
    return train_df, test_df


# ─────────────────────────────────────────────────────────────
# 4. BALANCEO DE CLASES (mejora #2): SMOTE + class_weight
# ─────────────────────────────────────────────────────────────
def _balancear_con_smote(X_train, y_train):
    """Aplica SMOTE sobre el set de entrenamiento. Si alguna clase
    minoritaria tiene menos muestras que k_neighbors+1, o si la
    librería no está disponible, se degrada de forma segura a
    devolver los datos originales (sin balancear) y se sigue
    confiando en class_weight="balanced" dentro del RandomForest."""
    if not _SMOTE_DISPONIBLE:
        return X_train, y_train, False

    conteo_clases = y_train.value_counts()
    min_clase = conteo_clases.min()
    k_neighbors = min(5, max(1, min_clase - 1))

    if min_clase < 2:
        # SMOTE no puede generar vecinos sintéticos con 1 sola muestra
        return X_train, y_train, False

    try:
        smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
        X_bal, y_bal = smote.fit_resample(X_train, y_train)
        return X_bal, y_bal, True
    except Exception:
        return X_train, y_train, False


# ─────────────────────────────────────────────────────────────
# 5. ENTRENAMIENTO + EVALUACIÓN COMPLETA
# ─────────────────────────────────────────────────────────────
def entrenar_modelo_riesgo(data_dir=DATA_DIR_DEFAULT, aplicar_smote=True, verbose=True):
    """Entrena el modelo de riesgo y devuelve un diccionario con todo
    lo necesario para servir predicciones (Flask) o exportar
    resultados (script standalone), incluyendo métricas detalladas
    por clase (mejora #5)."""

    inventario, ventas, rotacion = cargar_datos(data_dir)
    dataset = construir_dataset(inventario, ventas, rotacion)

    dataset["nivel_riesgo"] = dataset.apply(clasificar_riesgo, axis=1)
    dataset_train = dataset[dataset["nivel_riesgo"] != "VENCIDO"].copy()

    le_cat = LabelEncoder()
    dataset_train["categoria_enc"] = le_cat.fit_transform(dataset_train["categoria"])

    def encode_cat(cat):
        return le_cat.transform([cat])[0] if cat in le_cat.classes_ else 0

    train_df, test_df = _split_temporal_estratificado(dataset_train)

    X_train, y_train = train_df[FEATURES], train_df[TARGET]
    X_test, y_test = test_df[FEATURES], test_df[TARGET]

    X_train_bal, y_train_bal, smote_aplicado = (
        _balancear_con_smote(X_train, y_train) if aplicar_smote else (X_train, y_train, False)
    )

    # ── Modelo de evaluación (train temporal / test temporal) ──
    modelo_eval = RandomForestClassifier(
        n_estimators=100, max_depth=10, min_samples_leaf=2,
        random_state=42, class_weight="balanced",
    )
    modelo_eval.fit(X_train_bal, y_train_bal)
    y_pred_eval = modelo_eval.predict(X_test)

    clases_presentes = [c for c in CLASES_ORDEN if c in y_test.unique()]

    metricas = {
        "exactitud_pct": round(accuracy_score(y_test, y_pred_eval) * 100, 2),
        "f1_macro_pct": round(f1_score(y_test, y_pred_eval, average="macro", zero_division=0) * 100, 2),
        "f1_ponderado_pct": round(f1_score(y_test, y_pred_eval, average="weighted", zero_division=0) * 100, 2),
        "n_train": int(len(X_train)),
        "n_train_balanceado": int(len(X_train_bal)),
        "n_test": int(len(X_test)),
        "smote_aplicado": bool(smote_aplicado),
        "smote_disponible": bool(_SMOTE_DISPONIBLE),
        "por_clase": {},
        "matriz_confusion": {
            "clases": clases_presentes,
            "valores": confusion_matrix(y_test, y_pred_eval, labels=clases_presentes).tolist(),
        },
    }

    precision_c = precision_score(y_test, y_pred_eval, labels=clases_presentes, average=None, zero_division=0)
    recall_c = recall_score(y_test, y_pred_eval, labels=clases_presentes, average=None, zero_division=0)
    f1_c = f1_score(y_test, y_pred_eval, labels=clases_presentes, average=None, zero_division=0)
    soporte = y_test.value_counts()

    for i, clase in enumerate(clases_presentes):
        metricas["por_clase"][clase] = {
            "precision_pct": round(float(precision_c[i]) * 100, 2),
            "recall_pct": round(float(recall_c[i]) * 100, 2),
            "f1_pct": round(float(f1_c[i]) * 100, 2),
            "soporte": int(soporte.get(clase, 0)),
        }

    reporte_texto = classification_report(y_test, y_pred_eval, zero_division=0)

    if verbose:
        print(f"Evaluación (test temporal estratificado, {len(test_df)} muestras): "
              f"exactitud={metricas['exactitud_pct']}%  f1-macro={metricas['f1_macro_pct']}%  "
              f"SMOTE={'aplicado' if smote_aplicado else 'no aplicado (fallback class_weight)'}")
        print(reporte_texto)

    # ── Modelo de producción: se re-entrena con TODOS los datos ──
    X_full_bal, y_full_bal, _ = (
        _balancear_con_smote(dataset_train[FEATURES], dataset_train[TARGET])
        if aplicar_smote else (dataset_train[FEATURES], dataset_train[TARGET], False)
    )
    modelo = RandomForestClassifier(
        n_estimators=100, max_depth=10, min_samples_leaf=2,
        random_state=42, class_weight="balanced",
    )
    modelo.fit(X_full_bal, y_full_bal)

    medianas_categoria = dataset_train.groupby("categoria")[FEATURES[1:]].median()
    medianas_globales = dataset_train[FEATURES[1:]].median()

    # Predicciones sobre el inventario actual (último lote por producto)
    inventario_actual = dataset.sort_values("fecha_ingreso").groupby("id_producto").last().reset_index()
    inventario_actual["categoria_enc"] = inventario_actual["categoria"].apply(encode_cat)

    X_actual = inventario_actual[FEATURES].fillna(0)
    inventario_actual["prediccion_riesgo"] = modelo.predict(X_actual)
    proba = modelo.predict_proba(X_actual)
    inventario_actual["confianza_pct"] = (proba.max(axis=1) * 100).round(1)

    orden = {"CRITICO": 0, "ALTO": 1, "MODERADO": 2, "BAJO": 3, "VENCIDO": 4}
    inventario_actual["orden"] = inventario_actual["prediccion_riesgo"].map(orden)
    inventario_actual = inventario_actual.sort_values("orden")

    importancias = list(zip(FEATURES, modelo.feature_importances_))
    importancias.sort(key=lambda x: x[1], reverse=True)

    return {
        "modelo": modelo,
        "le_cat": le_cat,
        "encode_cat": encode_cat,
        "dataset": dataset,
        "dataset_train": dataset_train,
        "inventario_actual": inventario_actual,
        "medianas_categoria": medianas_categoria,
        "medianas_globales": medianas_globales,
        "importancias": importancias,
        "metricas": metricas,
        "reporte_texto": reporte_texto,
        "features": FEATURES,
    }


# ─────────────────────────────────────────────────────────────
# 6. EJECUCIÓN COMO SCRIPT — exporta a resultados/
# ─────────────────────────────────────────────────────────────
def _exportar_resultados(resultado, resultados_dir=RESULTADOS_DIR_DEFAULT):
    os.makedirs(resultados_dir, exist_ok=True)

    columnas_salida = [
        "id_producto", "nombre_producto", "categoria",
        "stock_disponible", "dias_para_vencer",
        "prediccion_riesgo", "confianza_pct",
    ]
    salida = resultado["inventario_actual"][columnas_salida]
    ruta_csv = os.path.join(resultados_dir, "predicciones_riesgo_vencimiento.csv")
    salida.to_csv(ruta_csv, index=False)

    ruta_json = os.path.join(resultados_dir, "metricas_evaluacion.json")
    with open(ruta_json, "w", encoding="utf-8") as f:
        json.dump(resultado["metricas"], f, ensure_ascii=False, indent=2)

    return ruta_csv, ruta_json


if __name__ == "__main__":
    print("=" * 60)
    print("  MARKET LAYA — Clasificación de Riesgo de Vencimiento")
    print("=" * 60)

    resultado = entrenar_modelo_riesgo(verbose=True)

    print("\nImportancia de variables:")
    for var, imp in resultado["importancias"]:
        barra = "█" * int(imp * 40)
        print(f"    {var:25s} {imp:.4f}  {barra}")

    ruta_csv, ruta_json = _exportar_resultados(resultado)

    print(f"\nResultados exportados:")
    print(f"  → {ruta_csv}")
    print(f"  → {ruta_json}")

    m = resultado["metricas"]
    print("\n" + "=" * 60)
    print("  RESUMEN FINAL")
    print("=" * 60)
    print(f"  Exactitud       : {m['exactitud_pct']}%")
    print(f"  F1-macro        : {m['f1_macro_pct']}%")
    print(f"  F1-ponderado    : {m['f1_ponderado_pct']}%")
    print(f"  SMOTE aplicado  : {m['smote_aplicado']}")
    for clase, met in m["por_clase"].items():
        print(f"  {clase:10s}: precision={met['precision_pct']}%  recall={met['recall_pct']}%  "
              f"f1={met['f1_pct']}%  soporte={met['soporte']}")
    print("=" * 60)
