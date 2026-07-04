# -*- coding: utf-8 -*-
"""
=============================================================
  MARKET LAYA — Chatbot de Consultas de Inventario
  Punto 6 de la consigna: Solución de IA (PLN + Sistema Conversacional)
=============================================================
Qué hace:
  1. CLASIFICACIÓN DE TEXTO (PLN): un modelo TF-IDF + Regresión
     Logística clasifica cada mensaje del usuario en una de 9
     intenciones (saludo, consultar_producto, listar_criticos,
     cuanto_pedir, etc.)
  2. EXTRACCIÓN DE ENTIDADES: busca, dentro del mensaje, el nombre
     de un producto o una categoría del catálogo de Market Laya
     (coincidencia difusa, tolera errores de tipeo).
  3. GENERACIÓN DE RESPUESTA: arma la respuesta con datos REALES.

  AJUSTE respecto a la versión anterior (resuelve inconsistencia #4):
  Antes este módulo leía, al importarse, un CSV congelado
  (predicciones_riesgo_vencimiento.csv) generado la última vez que
  se corrió modelo_ia_market_laya.py. Si alguien reentrenaba el
  modelo o cambiaba datos y solo reiniciaba app.py, el chatbot podía
  quedar respondiendo con predicciones desactualizadas -inconsistentes
  con lo que mostraba el dashboard-, porque cada uno leía una fuente
  distinta.

  Ahora responder() NO lee ningún CSV de resultados: recibe como
  parámetro el DataFrame de predicciones que app.py ya tiene en
  memoria (el mismo que sirve /datos), y opcionalmente el de
  recomendaciones de reposición (el mismo que sirve /reponer). Como
  Flask y el chatbot corren en el mismo proceso, esto garantiza que
  el chatbot y el dashboard siempre estén mirando exactamente los
  mismos números, sin necesidad de una llamada HTTP de ida y vuelta.

  MEJORA #6 — Nuevo intent: "cuanto debo pedir de X" → llama a las
  recomendaciones de modelo_reposicion.py en vez de solo hablar de
  riesgo de vencimiento.

  El catálogo de nombres/categorías para la extracción de entidades
  SÍ se carga una sola vez desde data/inventario_limpio.csv al
  importar el módulo — esto es intencional: el catálogo de productos
  (qué productos existen) es información de referencia estable, muy
  distinta a las predicciones (que sí cambian con cada reentrenamiento).

Requiere: pandas, scikit-learn
Ejecutar suelto para probarlo por consola (usa datos de ejemplo del
propio inventario, sin levantar Flask):
    python chatbot/chatbot.py
=============================================================
"""

import os
import re
import difflib

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR_DEFAULT = os.path.join(BASE_DIR, "data")

# ─────────────────────────────────────────────────────────────
# 1. CATÁLOGO DE REFERENCIA (estable, no son predicciones)
# ─────────────────────────────────────────────────────────────
def _cargar_catalogo(data_dir=DATA_DIR_DEFAULT):
    inv = pd.read_csv(os.path.join(data_dir, "inventario_limpio.csv"))
    nombres = sorted(inv["nombre_producto"].unique().tolist())
    categorias = sorted(inv["categoria"].unique().tolist())
    return nombres, categorias


CATALOGO_NOMBRES, CATALOGO_CATEGORIAS = _cargar_catalogo()

ACCIONES = {
    "CRITICO":  "aplicar descuento inmediato o coordinar devolución al proveedor",
    "ALTO":     "lanzar una promoción esta semana y reducir el próximo pedido",
    "MODERADO": "monitorear ventas; si no mejora en unos días, pasa a riesgo ALTO",
    "BAJO":     "operación normal, no requiere acción por ahora",
}

# ─────────────────────────────────────────────────────────────
# 2. CLASIFICACIÓN DE INTENCIÓN (PLN — TF-IDF + Regresión Logística)
# ─────────────────────────────────────────────────────────────
EJEMPLOS_ENTRENAMIENTO = [
    # saludo
    ("hola", "saludo"), ("buenas", "saludo"), ("buenos dias", "saludo"),
    ("hey", "saludo"), ("que tal", "saludo"), ("buenas tardes", "saludo"),

    # despedida
    ("gracias", "despedida"), ("muchas gracias", "despedida"), ("chau", "despedida"),
    ("adios", "despedida"), ("eso es todo gracias", "despedida"), ("nos vemos", "despedida"),

    # ayuda
    ("que puedes hacer", "ayuda"), ("ayuda", "ayuda"), ("como funciona esto", "ayuda"),
    ("que preguntas puedo hacer", "ayuda"), ("no se que preguntar", "ayuda"),
    ("ayudame", "ayuda"), ("necesito ayuda", "ayuda"), ("ayudaaa", "ayuda"),

    # consultar_producto (riesgo de un producto específico)
    ("como esta el queso fresco", "consultar_producto"),
    ("el yogurt laive esta en riesgo", "consultar_producto"),
    ("cuantos dias le quedan a la leche gloria", "consultar_producto"),
    ("dime el riesgo del arroz costeno", "consultar_producto"),
    ("que nivel de riesgo tiene la mantequilla", "consultar_producto"),
    ("revisa el estado de coca cola", "consultar_producto"),
    ("cual es el riesgo de vencimiento del aceite primor", "consultar_producto"),
    ("como va el papel higienico elite", "consultar_producto"),
    ("que tal esta la mantequilla laive", "consultar_producto"),
    ("que tal va el queso fresco", "consultar_producto"),

    # listar_criticos
    ("que productos estan en riesgo critico", "listar_criticos"),
    ("dame los productos criticos", "listar_criticos"),
    ("que hay que sacar ya del anaquel", "listar_criticos"),
    ("muestrame los productos urgentes", "listar_criticos"),
    ("hay algo en riesgo critico hoy", "listar_criticos"),
    ("necesito la lista de productos en peligro", "listar_criticos"),

    # listar_por_nivel (cualquier nivel de riesgo: bajo, moderado, alto)
    ("que productos tienen riesgo bajo", "listar_por_nivel"),
    ("que productos no necesitan atencion", "listar_por_nivel"),
    ("que productos no requieren atencion", "listar_por_nivel"),
    ("dame los productos en riesgo moderado", "listar_por_nivel"),
    ("que productos estan en riesgo alto", "listar_por_nivel"),
    ("cuales productos estan bien", "listar_por_nivel"),
    ("que productos estan seguros", "listar_por_nivel"),
    ("dame los productos sin riesgo", "listar_por_nivel"),
    ("que productos tienen nivel bajo", "listar_por_nivel"),
    ("que productos tienen riesgo moderado", "listar_por_nivel"),
    ("muestrame los productos en riesgo alto", "listar_por_nivel"),
    ("que productos estan en operacion normal", "listar_por_nivel"),

    # listar_por_categoria
    ("que lacteos estan en riesgo", "listar_por_categoria"),
    ("muestrame los productos de bebidas", "listar_por_categoria"),
    ("como estan los snacks", "listar_por_categoria"),
    ("dame el estado de la categoria limpieza", "listar_por_categoria"),
    ("que tal los productos de higiene", "listar_por_categoria"),

    # explicar_riesgo (por qué)
    ("por que el queso fresco esta en riesgo critico", "explicar_riesgo"),
    ("explicame por que el yogurt esta en moderado", "explicar_riesgo"),
    ("por que ese producto es riesgoso", "explicar_riesgo"),
    ("cual es la razon del riesgo del queso", "explicar_riesgo"),

    # recomendacion_general
    ("que debo hacer hoy", "recomendacion_general"),
    ("dame recomendaciones", "recomendacion_general"),
    ("que acciones debo tomar esta semana", "recomendacion_general"),
    ("cual es el resumen del dia", "recomendacion_general"),
    ("dame un resumen general del inventario", "recomendacion_general"),

    # cuanto_pedir (NUEVO — mejora #6, consulta al modelo_reposicion)
    ("cuanto debo pedir de queso fresco", "cuanto_pedir"),
    ("cuanto pedir de yogurt laive", "cuanto_pedir"),
    ("cuantas unidades debo comprar de arroz costeno", "cuanto_pedir"),
    ("que cantidad pido de leche gloria", "cuanto_pedir"),
    ("cuanto stock debo reponer de coca cola", "cuanto_pedir"),
    ("cuanto necesito pedir de aceite primor", "cuanto_pedir"),
    ("recomiendame cuanto comprar de papel higienico", "cuanto_pedir"),
    ("cuanto deberia reponer esta semana de arroz", "cuanto_pedir"),
    ("dame la sugerencia de pedido de yogurt", "cuanto_pedir"),
    ("que productos necesito reponer", "cuanto_pedir"),
]

textos_train = [t for t, _ in EJEMPLOS_ENTRENAMIENTO]
labels_train = [l for _, l in EJEMPLOS_ENTRENAMIENTO]

clasificador_intencion = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
    ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", C=8.0)),
])
clasificador_intencion.fit(textos_train, labels_train)


def normalizar(texto):
    texto = texto.lower().strip()
    texto = re.sub(r"[^\w\sáéíóúñ]", "", texto)
    return texto


def clasificar_intencion(mensaje):
    texto = normalizar(mensaje)
    proba = clasificador_intencion.predict_proba([texto])[0]
    clases = clasificador_intencion.classes_
    idx = proba.argmax()
    return clases[idx], round(proba[idx] * 100, 1)


# ─────────────────────────────────────────────────────────────
# 3. EXTRACCIÓN DE ENTIDADES: producto y/o categoría mencionados
# ─────────────────────────────────────────────────────────────
def extraer_producto(mensaje):
    texto = normalizar(mensaje)
    palabras = texto.split()
    mejor_match, mejor_score = None, 0.0

    for nombre in CATALOGO_NOMBRES:
        nombre_norm = normalizar(nombre)
        if nombre_norm in texto or any(w in nombre_norm for w in palabras if len(w) > 3):
            score = difflib.SequenceMatcher(None, nombre_norm, texto).ratio()
            if score > mejor_score:
                mejor_match, mejor_score = nombre, score

    if mejor_match is None:
        candidatos = difflib.get_close_matches(texto, [normalizar(n) for n in CATALOGO_NOMBRES], n=1, cutoff=0.4)
        if candidatos:
            idx = [normalizar(n) for n in CATALOGO_NOMBRES].index(candidatos[0])
            mejor_match = CATALOGO_NOMBRES[idx]

    return mejor_match


def extraer_categoria(mensaje):
    texto = normalizar(mensaje)
    for cat in CATALOGO_CATEGORIAS:
        if normalizar(cat) in texto:
            return cat
    return None


def extraer_nivel_riesgo(mensaje):
    """Detecta qué nivel de riesgo pidió el usuario, por palabras clave."""
    texto = normalizar(mensaje)
    if any(p in texto for p in ["no necesitan atencion", "no requieren atencion", "sin riesgo",
                                 "estan bien", "estan seguros", "operacion normal", "riesgo bajo",
                                 "nivel bajo"]) or " bajo" in f" {texto}":
        return "BAJO"
    if "moderado" in texto:
        return "MODERADO"
    if "alto" in texto and "critico" not in texto:
        return "ALTO"
    if any(p in texto for p in ["critico", "urgente", "peligro"]):
        return "CRITICO"
    return None


# ─────────────────────────────────────────────────────────────
# 4. GENERACIÓN DE RESPUESTA por intención
#    Todas las funciones reciben df_riesgo / df_reposicion como
#    parámetro: son las predicciones EN VIVO que le pasa app.py,
#    nunca un archivo leído por el propio chatbot.
# ─────────────────────────────────────────────────────────────
def _fila_producto(df_riesgo, nombre):
    fila = df_riesgo[df_riesgo["nombre_producto"] == nombre]
    return fila.iloc[0] if len(fila) else None


def responder_consultar_producto(mensaje, df_riesgo, df_reposicion=None):
    nombre = extraer_producto(mensaje)
    if not nombre:
        return ("No identifiqué el producto en tu mensaje. ¿Puedes decirme el nombre "
                "tal como aparece en el catálogo? Ej: 'Queso Fresco 250g'.")
    fila = _fila_producto(df_riesgo, nombre)
    if fila is None:
        return f"No encontré predicciones vigentes para '{nombre}'."
    riesgo = fila["prediccion_riesgo"]
    conf = fila["confianza_pct"]
    dias = int(fila["dias_para_vencer"])
    stock = int(fila["stock_disponible"])
    accion = ACCIONES.get(riesgo, "")
    return (f"📦 {nombre} ({fila['categoria']}): riesgo {riesgo} (confianza {conf}%). "
            f"Quedan {dias} días para vencer y hay {stock} unidades en stock. "
            f"Sugerencia: {accion}.")


def responder_listar_criticos(mensaje, df_riesgo, df_reposicion=None):
    criticos = df_riesgo[df_riesgo["prediccion_riesgo"].isin(["CRITICO", "ALTO"])].sort_values("dias_para_vencer")
    if len(criticos) == 0:
        return "✅ Buenas noticias: no hay productos en riesgo CRÍTICO ni ALTO ahora mismo."
    partes = [f"{r['nombre_producto']} ({r['prediccion_riesgo']}, {int(r['dias_para_vencer'])} días)"
              for _, r in criticos.iterrows()]
    return "⚠️ Productos que requieren atención: " + "; ".join(partes) + "."


def responder_listar_por_categoria(mensaje, df_riesgo, df_reposicion=None):
    cat = extraer_categoria(mensaje)
    if not cat:
        return ("¿De qué categoría? Tengo: " + ", ".join(CATALOGO_CATEGORIAS) + ".")
    sub = df_riesgo[df_riesgo["categoria"] == cat].sort_values("dias_para_vencer")
    partes = [f"{r['nombre_producto']}: {r['prediccion_riesgo']}" for _, r in sub.iterrows()]
    return f"📋 Categoría {cat} ({len(sub)} productos): " + "; ".join(partes) + "."


def responder_listar_por_nivel(mensaje, df_riesgo, df_reposicion=None):
    """Lista productos de CUALQUIER nivel de riesgo (bajo, moderado, alto)
    que el usuario haya mencionado. Si no se detecta un nivel explícito,
    cae de vuelta al comportamiento de 'listar_criticos' (CRÍTICO+ALTO)."""
    nivel = extraer_nivel_riesgo(mensaje)

    if nivel is None:
        return responder_listar_criticos(mensaje, df_riesgo, df_reposicion)

    sub = df_riesgo[df_riesgo["prediccion_riesgo"] == nivel].sort_values("dias_para_vencer")
    iconos = {"BAJO": "✅", "MODERADO": "🟡", "ALTO": "🔴", "CRITICO": "⚠️"}
    icono = iconos.get(nivel, "📋")

    if len(sub) == 0:
        return f"{icono} No hay productos en nivel {nivel} en este momento."

    partes = [f"{r['nombre_producto']} ({int(r['dias_para_vencer'])} días)" for _, r in sub.iterrows()]
    return f"{icono} Productos en riesgo {nivel} ({len(sub)}): " + "; ".join(partes) + "."


def responder_explicar_riesgo(mensaje, df_riesgo, df_reposicion=None):
    nombre = extraer_producto(mensaje)
    if not nombre:
        return "¿De qué producto quieres que te explique el riesgo?"
    fila = _fila_producto(df_riesgo, nombre)
    if fila is None:
        return f"No encontré predicciones vigentes para '{nombre}'."
    riesgo = fila["prediccion_riesgo"]
    dias = int(fila["dias_para_vencer"])
    stock = int(fila["stock_disponible"])
    if riesgo == "CRITICO":
        motivo = f"le quedan solo {dias} días y/o su stock ({stock} uds) no se agotaría a tiempo según su rotación histórica"
    elif riesgo == "ALTO":
        motivo = f"le quedan {dias} días, un margen ajustado según su patrón de venta"
    elif riesgo == "MODERADO":
        motivo = f"le quedan {dias} días; todavía hay margen pero conviene vigilarlo"
    else:
        motivo = f"le quedan {dias} días, tiempo de sobra dado su ritmo de venta habitual"
    return f"🔎 {nombre} está en {riesgo} porque {motivo}."


def responder_recomendacion_general(mensaje, df_riesgo, df_reposicion=None):
    n_crit = len(df_riesgo[df_riesgo["prediccion_riesgo"] == "CRITICO"])
    n_alto = len(df_riesgo[df_riesgo["prediccion_riesgo"] == "ALTO"])
    n_mod = len(df_riesgo[df_riesgo["prediccion_riesgo"] == "MODERADO"])
    return (f"📊 Resumen de hoy: {n_crit} producto(s) en CRÍTICO, {n_alto} en ALTO y "
            f"{n_mod} en MODERADO. Prioriza los CRÍTICOS con descuento inmediato y revisa "
            f"los ALTOS para promoción esta semana.")


def responder_cuanto_pedir(mensaje, df_riesgo, df_reposicion=None):
    """NUEVO (mejora #6): consulta la recomendación de reposición
    generada por modelo_reposicion.py (recibida en vivo desde app.py)."""
    if df_reposicion is None or len(df_reposicion) == 0:
        return "No tengo disponibles las recomendaciones de reposición en este momento."

    nombre = extraer_producto(mensaje)
    if not nombre:
        # Sin producto específico → top 5 productos con mayor cantidad sugerida
        top = df_reposicion.sort_values("cantidad_sugerida_pedido", ascending=False).head(5)
        partes = [f"{r['nombre_producto']}: {r['cantidad_sugerida_pedido']} uds" for _, r in top.iterrows()]
        return ("¿De qué producto? Estos son los que más reposición necesitan ahora: "
                + "; ".join(partes) + ".")

    fila = df_reposicion[df_reposicion["nombre_producto"] == nombre]
    if len(fila) == 0:
        return f"No encontré una recomendación de reposición para '{nombre}'."
    r = fila.iloc[0]
    return (f"🛒 {nombre}: se estiman {r['ventas_estimadas_proxima_semana']:.0f} unidades vendidas "
            f"la próxima semana (stock actual: {r['stock_disponible']}). "
            f"Sugerencia de pedido: {int(r['cantidad_sugerida_pedido'])} unidades "
            f"[método: {r['metodo']}].")


RESPUESTAS_FIJAS = {
    "saludo": "¡Hola! Soy el asistente de Market Laya. Puedo decirte qué productos están "
              "en riesgo de vencer, explicarte por qué, cuánto pedir de reposición, o darte "
              "un resumen del inventario.",
    "despedida": "¡De nada! Aquí estaré si necesitas revisar el inventario de nuevo.",
    "ayuda": ("Puedo ayudarte con:\n"
              "  • 'cómo está el queso fresco' (consultar un producto)\n"
              "  • 'qué productos están en riesgo crítico' (lista urgente)\n"
              "  • 'qué productos tienen riesgo bajo/moderado/alto' (por nivel)\n"
              "  • 'qué lácteos están en riesgo' (por categoría)\n"
              "  • 'por qué el yogurt está en riesgo' (explicación)\n"
              "  • 'cuánto debo pedir de arroz costeño' (reposición)\n"
              "  • 'dame un resumen del día' (recomendación general)"),
}

MANEJADORES = {
    "consultar_producto":    responder_consultar_producto,
    "listar_criticos":       responder_listar_criticos,
    "listar_por_nivel":      responder_listar_por_nivel,
    "listar_por_categoria":  responder_listar_por_categoria,
    "explicar_riesgo":       responder_explicar_riesgo,
    "recomendacion_general": responder_recomendacion_general,
    "cuanto_pedir":          responder_cuanto_pedir,
}


def responder(mensaje, df_riesgo, df_reposicion=None):
    """Punto de entrada único: recibe un mensaje de texto libre y las
    predicciones EN VIVO (df_riesgo, df_reposicion) que ya tiene
    app.py en memoria, y devuelve la respuesta generada.

    df_riesgo requiere las columnas: nombre_producto, categoria,
    stock_disponible, dias_para_vencer, prediccion_riesgo, confianza_pct
    (el mismo formato que sirve el endpoint /datos).

    df_reposicion (opcional) requiere las columnas: nombre_producto,
    stock_disponible, ventas_estimadas_proxima_semana,
    cantidad_sugerida_pedido, metodo (el mismo formato que /reponer).
    """
    if not mensaje or not mensaje.strip():
        return {"intencion": None, "confianza": 0, "respuesta": "Escribe algo para que pueda ayudarte 🙂"}

    intencion, confianza = clasificar_intencion(mensaje)

    if intencion in RESPUESTAS_FIJAS:
        respuesta = RESPUESTAS_FIJAS[intencion]
    else:
        respuesta = MANEJADORES[intencion](mensaje, df_riesgo, df_reposicion)

    return {"intencion": intencion, "confianza": confianza, "respuesta": respuesta}


# ─────────────────────────────────────────────────────────────
# 5. PRUEBA POR CONSOLA (standalone, sin Flask)
#    Entrena rápidamente ambos modelos solo para tener datos con
#    los cuales probar el chatbot de forma aislada.
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, BASE_DIR)
    from modelos.modelo_riesgo import entrenar_modelo_riesgo
    from modelos.modelo_reposicion import entrenar_modelo_reposicion

    print("=" * 60)
    print("  MARKET LAYA — Chatbot de inventario (modo standalone)")
    print("=" * 60)
    print("Entrenando modelos para la demo...")

    res_riesgo = entrenar_modelo_riesgo(verbose=False)
    res_reposicion = entrenar_modelo_reposicion(verbose=False)
    df_riesgo_demo = res_riesgo["inventario_actual"]
    df_reposicion_demo = res_reposicion["recomendaciones"]

    ejemplos_demo = [
        "hola",
        "como esta el queso fresco",
        "que productos estan en riesgo critico",
        "que lacteos estan en riesgo",
        "por que el queso fresco esta en riesgo critico",
        "cuanto debo pedir de arroz costeno",
        "dame un resumen del dia",
        "gracias",
    ]
    print("\n[Demo automática con frases de ejemplo]\n")
    for msg in ejemplos_demo:
        r = responder(msg, df_riesgo_demo, df_reposicion_demo)
        print(f"👤 {msg}")
        print(f"🤖 [{r['intencion']} · {r['confianza']}%] {r['respuesta']}\n")

    print("=" * 60)
    print("Ahora puedes escribir tus propias preguntas (Ctrl+C para salir):")
    try:
        while True:
            msg = input("\n👤 > ")
            r = responder(msg, df_riesgo_demo, df_reposicion_demo)
            print(f"🤖 [{r['intencion']} · {r['confianza']}%] {r['respuesta']}")
    except (KeyboardInterrupt, EOFError):
        print("\n\nHasta luego 👋")
