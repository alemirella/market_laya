# -*- coding: utf-8 -*-
"""
=============================================================
  MARKET LAYA - Modelo de IA: Estimación de Reposición Semanal
  Asignatura : Inteligencia de Negocios
  Universidad: Universidad Continental
  Algoritmo  : Random Forest Regressor + regla de respaldo

  NUEVO MÓDULO — responde la pregunta que faltaba en la versión
  anterior del proyecto: "¿cuánto debería pedir la próxima semana
  de cada producto?". El modelo de riesgo (modelo_riesgo.py) solo
  clasifica el riesgo de vencimiento; no dice qué hacer respecto a
  cuánto reponer, que es la otra mitad del problema de negocio
  (evitar quiebres de stock sin sobre-comprar productos que ya
  están en riesgo de vencer).

  ENFOQUE (dos capas, con degradación segura):
  1) REGRESIÓN sobre la serie histórica semanal de ventas por
     producto (features: rezagos de ventas de las 3 semanas
     anteriores, categoría, costo, precio, tasa_rotacion,
     stock_promedio) para estimar las unidades que se venderán
     la semana próxima.
  2) Si un producto no tiene suficiente historial semanal para
     construir rezagos confiables (< 4 semanas de datos), se cae
     a una REGLA basada en tasa_rotacion vs. stock_promedio
     (consumo diario esperado = tasa_rotacion / 365; se usa esa
     tasa para proyectar el consumo de 7 días). Esto también sirve
     como validación cruzada de sentido común frente al modelo ML.

  La recomendación final de pedido combina la estimación de ventas
  de la próxima semana con el stock actual y un stock de seguridad:

      cantidad_sugerida = max(0, ventas_estimadas_semana * (1 + margen_seguridad)
                                  - stock_disponible_actual)

  Este script se puede ejecutar de forma independiente:
      python modelos/modelo_reposicion.py
  (ejecutar desde la raíz del proyecto, market_laya/), y exporta:
      resultados/recomendaciones_reposicion.csv
=============================================================
"""

import os
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR_DEFAULT = os.path.join(BASE_DIR, "data")
RESULTADOS_DIR_DEFAULT = os.path.join(BASE_DIR, "resultados")

N_REZAGOS = 3                 # semanas anteriores usadas como features
MIN_SEMANAS_PARA_ML = 4       # mínimo de semanas de historia para confiar en el modelo ML
MARGEN_SEGURIDAD = 0.15       # 15% de stock de seguridad sobre la venta estimada

FEATURES_ML = [
    "categoria_enc", "costo_unitario", "precio_venta_promedio",
    "tasa_rotacion", "stock_promedio",
] + [f"ventas_lag_{i}" for i in range(1, N_REZAGOS + 1)]


# ─────────────────────────────────────────────────────────────
# 1. CARGA Y AGREGACIÓN SEMANAL
# ─────────────────────────────────────────────────────────────
def cargar_datos(data_dir=DATA_DIR_DEFAULT):
    inventario = pd.read_csv(os.path.join(data_dir, "inventario_limpio.csv"))
    ventas = pd.read_csv(os.path.join(data_dir, "ventas_limpio.csv"))
    rotacion = pd.read_csv(os.path.join(data_dir, "rotacion_productos.csv"))
    return inventario, ventas, rotacion


def _agregar_ventas_semanales(ventas):
    """Agrega las ventas diarias en totales semanales por producto,
    usando semana ISO (lunes a domingo) para evitar mezclar meses
    parciales."""
    v = ventas.copy()
    v["fecha"] = pd.to_datetime(v["fecha"])
    v["semana"] = v["fecha"].dt.to_period("W").apply(lambda p: p.start_time)
    semanal = (
        v.groupby(["id_producto", "semana"])["cantidad_vendida"]
        .sum().reset_index()
        .rename(columns={"cantidad_vendida": "ventas_semana"})
        .sort_values(["id_producto", "semana"])
    )
    return semanal


def construir_dataset_reposicion(inventario, ventas, rotacion):
    """Construye, por producto y semana, los rezagos de ventas y las
    variables de contexto (categoría, costo, precio, rotación,
    stock_promedio), junto con el target (ventas de esa semana)."""
    semanal = _agregar_ventas_semanales(ventas)

    rotacion_sel = rotacion[["id_producto", "tasa_rotacion", "stock_promedio"]].copy()
    precio_promedio = (
        ventas.groupby("id_producto")["precio_venta"].mean().reset_index()
        .rename(columns={"precio_venta": "precio_venta_promedio"})
    )
    costo_ref = (
        inventario.sort_values("fecha_ingreso")
        .groupby("id_producto")[["costo_unitario", "categoria", "nombre_producto"]].last()
        .reset_index()
    )

    contexto = costo_ref.merge(rotacion_sel, on="id_producto", how="left")
    contexto = contexto.merge(precio_promedio, on="id_producto", how="left")

    for col in ["tasa_rotacion", "stock_promedio", "precio_venta_promedio", "costo_unitario"]:
        contexto[col] = contexto[col].fillna(contexto[col].median())

    le_cat = LabelEncoder()
    contexto["categoria_enc"] = le_cat.fit_transform(contexto["categoria"])

    filas = []
    for id_producto, grupo in semanal.groupby("id_producto"):
        grupo = grupo.sort_values("semana").reset_index(drop=True)
        serie = grupo["ventas_semana"].tolist()
        semanas = grupo["semana"].tolist()

        if id_producto not in contexto["id_producto"].values:
            continue
        ctx = contexto[contexto["id_producto"] == id_producto].iloc[0]

        for i in range(N_REZAGOS, len(serie)):
            fila = {
                "id_producto": id_producto,
                "semana": semanas[i],
                "target_ventas_semana": serie[i],
            }
            for lag in range(1, N_REZAGOS + 1):
                fila[f"ventas_lag_{lag}"] = serie[i - lag]
            for col in ["categoria_enc", "costo_unitario", "precio_venta_promedio",
                        "tasa_rotacion", "stock_promedio"]:
                fila[col] = ctx[col]
            filas.append(fila)

    dataset_ml = pd.DataFrame(filas)
    return dataset_ml, semanal, contexto, le_cat


# ─────────────────────────────────────────────────────────────
# 2. ENTRENAMIENTO DEL MODELO ML (rezagos → ventas próxima semana)
# ─────────────────────────────────────────────────────────────
def _entrenar_modelo_ml(dataset_ml):
    dataset_ml = dataset_ml.sort_values("semana").reset_index(drop=True)
    corte = max(1, int(len(dataset_ml) * 0.8))
    train_df = dataset_ml.iloc[:corte]
    test_df = dataset_ml.iloc[corte:]

    modelo = RandomForestRegressor(
        n_estimators=200, max_depth=8, min_samples_leaf=2, random_state=42,
    )
    modelo.fit(train_df[FEATURES_ML], train_df["target_ventas_semana"])

    metricas_ml = {"n_train": int(len(train_df)), "n_test": int(len(test_df))}
    if len(test_df) > 0:
        y_pred = modelo.predict(test_df[FEATURES_ML])
        metricas_ml["mae"] = round(float(mean_absolute_error(test_df["target_ventas_semana"], y_pred)), 2)
        metricas_ml["r2"] = round(float(r2_score(test_df["target_ventas_semana"], y_pred)), 3) if len(test_df) > 1 else None
    else:
        metricas_ml["mae"] = None
        metricas_ml["r2"] = None

    # Re-entrenar con todos los datos disponibles para producción
    modelo_prod = RandomForestRegressor(
        n_estimators=200, max_depth=8, min_samples_leaf=2, random_state=42,
    )
    modelo_prod.fit(dataset_ml[FEATURES_ML], dataset_ml["target_ventas_semana"])

    return modelo_prod, metricas_ml


# ─────────────────────────────────────────────────────────────
# 3. REGLA DE RESPALDO (tasa_rotacion vs stock_promedio)
# ─────────────────────────────────────────────────────────────
def _estimar_por_regla(tasa_rotacion, stock_promedio):
    """Estimación simple de ventas semanales cuando no hay historial
    suficiente para el modelo ML: se proyecta el consumo diario
    promedio (tasa_rotacion / 365 días) a 7 días."""
    tasa = tasa_rotacion if tasa_rotacion and tasa_rotacion > 0 else 0
    return round((tasa / 365.0) * 7, 2)


# ─────────────────────────────────────────────────────────────
# 4. GENERAR RECOMENDACIONES PARA EL INVENTARIO ACTUAL
# ─────────────────────────────────────────────────────────────
def entrenar_modelo_reposicion(data_dir=DATA_DIR_DEFAULT, verbose=True):
    inventario, ventas, rotacion = cargar_datos(data_dir)
    dataset_ml, semanal, contexto, le_cat = construir_dataset_reposicion(inventario, ventas, rotacion)

    modelo_ml, metricas_ml = _entrenar_modelo_ml(dataset_ml) if len(dataset_ml) >= 20 else (None, {"n_train": 0, "n_test": 0, "mae": None, "r2": None})

    if verbose:
        print(f"Dataset de reposición: {len(dataset_ml)} observaciones semana-producto "
              f"({dataset_ml['id_producto'].nunique() if len(dataset_ml) else 0} productos con historial suficiente)")
        if modelo_ml is not None:
            print(f"Modelo ML entrenado — MAE={metricas_ml['mae']}  R²={metricas_ml['r2']}  "
                  f"(train={metricas_ml['n_train']}, test={metricas_ml['n_test']})")
        else:
            print("Historial insuficiente para el modelo ML — se usará la regla de respaldo para todos los productos.")

    inventario_actual = (
        inventario.sort_values("fecha_ingreso").groupby("id_producto").last().reset_index()
    )

    filas_salida = []
    for _, ctx in contexto.iterrows():
        id_producto = ctx["id_producto"]
        grupo = semanal[semanal["id_producto"] == id_producto].sort_values("semana")
        n_semanas = len(grupo)

        usa_ml = modelo_ml is not None and n_semanas >= (N_REZAGOS + MIN_SEMANAS_PARA_ML)

        if usa_ml:
            ultimos = grupo["ventas_semana"].tolist()[-N_REZAGOS:]
            fila_x = {f"ventas_lag_{lag}": ultimos[-lag] for lag in range(1, N_REZAGOS + 1)}
            for col in ["categoria_enc", "costo_unitario", "precio_venta_promedio",
                        "tasa_rotacion", "stock_promedio"]:
                fila_x[col] = ctx[col]
            X_pred = pd.DataFrame([fila_x])[FEATURES_ML]
            ventas_estimadas = max(0.0, float(modelo_ml.predict(X_pred)[0]))
            metodo = "ML (Random Forest)"
        else:
            ventas_estimadas = _estimar_por_regla(ctx["tasa_rotacion"], ctx["stock_promedio"])
            metodo = "Regla (tasa_rotacion vs stock)"

        fila_inv = inventario_actual[inventario_actual["id_producto"] == id_producto]
        stock_actual = int(fila_inv["stock_disponible"].iloc[0]) if len(fila_inv) else 0

        cantidad_sugerida = max(0, round(ventas_estimadas * (1 + MARGEN_SEGURIDAD) - stock_actual))

        filas_salida.append({
            "id_producto": id_producto,
            "nombre_producto": ctx["nombre_producto"],
            "categoria": ctx["categoria"],
            "stock_disponible": stock_actual,
            "ventas_semanales_promedio": round(float(grupo["ventas_semana"].mean()), 2) if n_semanas else 0.0,
            "ventas_estimadas_proxima_semana": round(ventas_estimadas, 2),
            "cantidad_sugerida_pedido": int(cantidad_sugerida),
            "metodo": metodo,
            "semanas_historial": int(n_semanas),
        })

    recomendaciones = pd.DataFrame(filas_salida).sort_values(
        "cantidad_sugerida_pedido", ascending=False
    ).reset_index(drop=True)

    return {
        "modelo_ml": modelo_ml,
        "metricas_ml": metricas_ml,
        "recomendaciones": recomendaciones,
        "dataset_ml": dataset_ml,
        "contexto": contexto,
        "le_cat": le_cat,
    }


def recomendar_producto(resultado_reposicion, nombre_o_id):
    """Utilidad para el chatbot: busca la recomendación de un producto
    por nombre o id_producto (coincidencia exacta o parcial)."""
    df = resultado_reposicion["recomendaciones"]
    coincidencia = df[
        (df["id_producto"] == nombre_o_id)
        | (df["nombre_producto"].str.lower() == str(nombre_o_id).lower())
    ]
    if len(coincidencia) == 0:
        coincidencia = df[df["nombre_producto"].str.lower().str.contains(str(nombre_o_id).lower(), na=False)]
    if len(coincidencia) == 0:
        return None
    return coincidencia.iloc[0]


# ─────────────────────────────────────────────────────────────
# 5. EJECUCIÓN COMO SCRIPT — exporta a resultados/
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  MARKET LAYA — Estimación de Reposición Semanal")
    print("=" * 60)

    resultado = entrenar_modelo_reposicion(verbose=True)
    recomendaciones = resultado["recomendaciones"]

    print(f"\n{'ID':6s} {'Producto':28s} {'Stock':6s} {'Est.sem':8s} {'Pedir':6s} {'Método':28s}")
    print("-" * 90)
    for _, r in recomendaciones.iterrows():
        print(f"{r['id_producto']:6s} {r['nombre_producto']:28s} {r['stock_disponible']:6d} "
              f"{r['ventas_estimadas_proxima_semana']:8.1f} {r['cantidad_sugerida_pedido']:6d} {r['metodo']:28s}")

    os.makedirs(RESULTADOS_DIR_DEFAULT, exist_ok=True)
    ruta_csv = os.path.join(RESULTADOS_DIR_DEFAULT, "recomendaciones_reposicion.csv")
    recomendaciones.to_csv(ruta_csv, index=False)

    print(f"\nResultados exportados → {ruta_csv}")
    print("=" * 60)
