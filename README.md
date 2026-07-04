# Market Laya — Sistema de IA para gestión de inventario

Sistema de apoyo a la decisión para Market Laya: clasifica el riesgo
de vencimiento del inventario y estima cuánto reponer la próxima
semana, con un dashboard web y un chatbot de consultas en lenguaje
natural.

## Estructura del proyecto

```
market_laya/
│
├── data/                              Datos limpios de entrada
│   ├── inventario_limpio.csv
│   ├── ventas_limpio.csv
│   └── rotacion_productos.csv
│
├── modelos/
│   ├── modelo_riesgo.py               Clasificación de riesgo (Random Forest)
│   └── modelo_reposicion.py           Estimación de reposición semanal (NUEVO)
│
├── resultados/                        Salidas generadas por los scripts
│   ├── predicciones_riesgo_vencimiento.csv
│   ├── metricas_evaluacion.json
│   └── recomendaciones_reposicion.csv
│
├── chatbot/
│   └── chatbot.py                     Asistente conversacional (PLN)
│
├── templates/
│   └── index.html                     Dashboard
│
├── evaluacion/
│   └── evaluacion_modelos.py          Validación cruzada, matriz de confusión, comparación temporal
│
├── docs/
│   └── limitaciones_trabajo_futuro.md
│
├── app.py                             Servidor Flask
└── requirements.txt
```

## Instalación

```bash
pip install -r requirements.txt
```

## Uso

### 1) Entrenar y exportar resultados por separado (opcional)

```bash
python modelos/modelo_riesgo.py        # → resultados/predicciones_riesgo_vencimiento.csv
                                        #   resultados/metricas_evaluacion.json
python modelos/modelo_reposicion.py    # → resultados/recomendaciones_reposicion.csv
python evaluacion/evaluacion_modelos.py  # → resultados/matriz_confusion.png
                                          #   resultados/comparacion_temporal.png
```

### 2) Levantar el dashboard (entrena ambos modelos al arrancar)

```bash
python app.py
```

Luego abrir `http://localhost:5000`.

## Endpoints del servidor

| Endpoint      | Método | Descripción |
|---------------|--------|-------------|
| `/`           | GET    | Dashboard |
| `/datos`      | GET    | Predicciones de riesgo vigentes + importancia de variables + KPIs |
| `/metricas`   | GET    | Métricas detalladas por clase (precision/recall/F1) y matriz de confusión |
| `/categorias` | GET    | Categorías disponibles |
| `/predecir`   | POST   | Clasifica el riesgo de un producto nuevo (simulación) |
| `/reponer`    | GET    | Recomendaciones de reposición semanal (`?producto=` opcional) |
| `/tendencia`  | GET    | Evolución mensual de la distribución de niveles de riesgo |
| `/chat`       | POST   | Chatbot de consultas en lenguaje natural |

## Notas de metodología

Ver `docs/limitaciones_trabajo_futuro.md` para el detalle de lo que
no se alcanzó a implementar y las razones de cada limitación.
