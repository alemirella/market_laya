"""
Market Laya — Backend Flask
Entrena AMBOS modelos al arrancar (riesgo + reposición) y expone
los endpoints para el dashboard y el chatbot.
Correr con: python app.py   (desde la raíz del proyecto, market_laya/)
Luego abrir: http://localhost:5000

ESTRUCTURA (ver README / árbol del proyecto):
  modelos/modelo_riesgo.py       → clasificación de riesgo de vencimiento
  modelos/modelo_reposicion.py   → estimación de cuánto pedir la próxima semana
  chatbot/chatbot.py             → asistente conversacional (consulta datos EN VIVO)
  evaluacion/evaluacion_modelos.py → evaluación separada del entrenamiento (CV, matrices)
  resultados/                    → CSV/JSON exportados por los scripts standalone

Este servidor reutiliza exactamente la misma lógica de entrenamiento
que los scripts standalone (modelos/modelo_riesgo.py y
modelos/modelo_reposicion.py) importando sus funciones, en vez de
duplicar el feature engineering aquí. Así el dashboard, el chatbot y
los resultados exportados en resultados/ nunca pueden desincronizarse
entre sí.
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from flask import Flask, render_template, jsonify, request
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from modelos.modelo_riesgo import entrenar_modelo_riesgo
from modelos.modelo_reposicion import entrenar_modelo_reposicion
from chatbot.chatbot import responder  # chatbot de consultas (Punto 6, PLN + intenciones)

app = Flask(__name__)

# ─────────────────────────────────────────────────────────────
# 1. ENTRENAR AMBOS MODELOS AL ARRANCAR
# ─────────────────────────────────────────────────────────────
print("Cargando datos y entrenando modelo de RIESGO...")
res_riesgo = entrenar_modelo_riesgo(verbose=True)

print("\nCargando datos y entrenando modelo de REPOSICIÓN...")
res_reposicion = entrenar_modelo_reposicion(verbose=True)

modelo = res_riesgo["modelo"]
FEATURES = res_riesgo["features"]
dataset = res_riesgo["dataset"]
dataset_train = res_riesgo["dataset_train"]
inventario_actual = res_riesgo["inventario_actual"]
medianas_categoria = res_riesgo["medianas_categoria"]
medianas_globales = res_riesgo["medianas_globales"]
importancias = res_riesgo["importancias"]
metricas = res_riesgo["metricas"]
encode_cat = res_riesgo["encode_cat"]

recomendaciones_reposicion = res_reposicion["recomendaciones"]

print("\nServidor listo en http://localhost:5000")


# ─────────────────────────────────────────────────────────────
# 2. UTILIDAD — evolución de riesgo entre cortes de tiempo
#    (para /tendencia): a partir del historial completo de lotes
#    (no solo el inventario actual), se etiqueta cada lote con su
#    nivel de riesgo y se agrupa por mes de ingreso, para ver cómo
#    ha evolucionado la proporción de productos en cada nivel.
# ─────────────────────────────────────────────────────────────
def _calcular_tendencia_riesgo():
    hist = dataset_train.copy()
    hist["mes"] = hist["fecha_ingreso"].dt.to_period("M").astype(str)
    conteo = hist.groupby(["mes", "nivel_riesgo"]).size().unstack(fill_value=0)
    for col in ["CRITICO", "ALTO", "MODERADO", "BAJO"]:
        if col not in conteo.columns:
            conteo[col] = 0
    conteo = conteo[["CRITICO", "ALTO", "MODERADO", "BAJO"]].sort_index()

    return {
        "meses": conteo.index.tolist(),
        "series": {col: conteo[col].tolist() for col in conteo.columns},
    }


# ─────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/datos")
def datos():
    """Devuelve predicciones actuales + importancia de variables + métricas reales."""
    productos = []
    for _, r in inventario_actual.iterrows():
        productos.append({
            "id":        r["id_producto"],
            "nombre":    r["nombre_producto"],
            "categoria": r["categoria"],
            "stock":     int(r["stock_disponible"]),   # referencia, no es input del modelo
            "dias":      int(r["dias_para_vencer"]),   # referencia, no es input del modelo
            "riesgo":    r["prediccion_riesgo"],
            "confianza": float(r["confianza_pct"]),
        })

    vars_imp = [{"nombre": v, "pct": round(p * 100, 1)} for v, p in importancias]

    criticos  = sum(1 for p in productos if p["riesgo"] == "CRITICO")
    altos     = sum(1 for p in productos if p["riesgo"] == "ALTO")
    moderados = sum(1 for p in productos if p["riesgo"] == "MODERADO")

    return jsonify({
        "productos":    productos,
        "importancias": vars_imp,
        "kpis": {
            "criticos":        criticos,
            "altos":           altos,
            "moderados":       moderados,
            "total":           len(productos),
            "exactitud_pct":   metricas["exactitud_pct"],
            "f1_macro_pct":    metricas["f1_macro_pct"],
            "n_features":      len(FEATURES),
            "n_test":          metricas["n_test"],
            "smote_aplicado":  metricas["smote_aplicado"],
        }
    })


@app.route("/metricas")
def metricas_endpoint():
    """Métricas detalladas por clase (mejora #5): precision/recall/F1
    por nivel de riesgo y matriz de confusión, no solo accuracy global."""
    return jsonify(metricas)


@app.route("/categorias")
def categorias():
    """Lista las categorías disponibles, para poblar el <select> del formulario."""
    return jsonify(sorted(dataset_train["categoria"].unique().tolist()))


@app.route("/predecir", methods=["POST"])
def predecir():
    """
    Simula un producto NUEVO y devuelve la clasificación del modelo real.
    Pide solo lo que se conoce de antemano (categoría, costo, precio
    estimado). La rotación/ventas históricas -que un producto nuevo
    todavía no tiene- se aproximan con la mediana de su categoría.
    """
    data = request.get_json()

    nombre    = data.get("nombre") or "Producto nuevo"
    categoria = data.get("categoria", "Alimentos")
    costo     = data.get("costo_unitario")
    precio    = data.get("precio_venta")

    cat_enc = encode_cat(categoria)
    meds = medianas_categoria.loc[categoria] if categoria in medianas_categoria.index else medianas_globales

    costo_final  = float(costo) if costo not in (None, "") else float(meds["costo_unitario"])
    precio_final = float(precio) if precio not in (None, "") else float(meds["precio_venta_promedio"])

    X_nuevo = pd.DataFrame([[
        cat_enc,
        costo_final,
        meds["tasa_rotacion"],
        meds["unidades_vendidas"],
        meds["ventas_30d"],
        precio_final,
        meds["stock_promedio"],
    ]], columns=FEATURES)

    nivel     = modelo.predict(X_nuevo)[0]
    confianza = round(modelo.predict_proba(X_nuevo).max() * 100, 1)

    acciones = {
        "CRITICO":  "Aplicar descuento inmediato o coordinar devolución al proveedor.",
        "ALTO":     "Lanzar promoción esta semana y reducir el próximo pedido.",
        "MODERADO": "Monitorear ventas. Si no mejora en 3 días, escalar a ALTO.",
        "BAJO":     "Operación normal. Continuar con el plan de ventas habitual.",
    }

    return jsonify({
        "nombre":    nombre,
        "categoria": categoria,
        "nivel":     nivel,
        "confianza": confianza,
        "accion":    acciones.get(nivel, ""),
        "supuestos": {
            "costo_unitario":        round(costo_final, 2),
            "precio_venta":          round(precio_final, 2),
            "tasa_rotacion_est":     round(float(meds["tasa_rotacion"]), 2),
            "unidades_vendidas_est": round(float(meds["unidades_vendidas"]), 1),
        }
    })


@app.route("/reponer")
def reponer():
    """
    NUEVO — devuelve la recomendación de cuánto pedir la próxima semana
    para cada producto (modelo_reposicion.py). Acepta opcionalmente
    ?producto=<nombre o id> para filtrar un solo producto.
    """
    producto = request.args.get("producto")
    df = recomendaciones_reposicion

    if producto:
        df = df[
            (df["id_producto"] == producto)
            | (df["nombre_producto"].str.lower().str.contains(producto.lower(), na=False))
        ]

    filas = []
    for _, r in df.iterrows():
        filas.append({
            "id":                     r["id_producto"],
            "nombre":                 r["nombre_producto"],
            "categoria":              r["categoria"],
            "stock":                  int(r["stock_disponible"]),
            "ventas_semanales_prom":  float(r["ventas_semanales_promedio"]),
            "ventas_estimadas":       float(r["ventas_estimadas_proxima_semana"]),
            "cantidad_sugerida":      int(r["cantidad_sugerida_pedido"]),
            "metodo":                 r["metodo"],
        })

    return jsonify({
        "recomendaciones": filas,
        "metricas_modelo_ml": res_reposicion["metricas_ml"],
    })


@app.route("/tendencia")
def tendencia():
    """
    NUEVO (opcional) — evolución de la distribución de niveles de
    riesgo mes a mes, a partir del historial completo de lotes
    ingresados (no solo el inventario vigente).
    """
    return jsonify(_calcular_tendencia_riesgo())


@app.route("/chat", methods=["POST"])
def chat():
    """
    Endpoint del chatbot (Punto 6). Recibe un mensaje en lenguaje natural
    y devuelve la respuesta generada por chatbot.responder(), pasándole
    las predicciones EN VIVO (mismo inventario_actual que sirve /datos,
    mismas recomendaciones que sirve /reponer) — nunca un CSV congelado.
    """
    data = request.get_json(silent=True) or {}
    mensaje = data.get("mensaje", "")
    resultado = responder(mensaje, inventario_actual, recomendaciones_reposicion)
    return jsonify(resultado)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
