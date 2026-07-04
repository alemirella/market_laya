# Limitaciones y trabajo futuro — Market Laya

Este documento reúne, de forma explícita, lo que la solución actual
**no** alcanza a resolver, junto con las razones metodológicas o de
alcance del proyecto, y qué se recomendaría abordar en una siguiente
iteración. Su propósito es servir de insumo directo para la sección
de limitaciones y trabajo futuro del informe/tesis.

## 1. Tamaño y representatividad de los datos

- El catálogo consta de **25 productos** y ~800 lotes de inventario.
  Es un volumen adecuado para un proyecto académico, pero insuficiente
  para que un Random Forest generalice con solidez estadística en las
  clases minoritarias (CRITICO, ALTO): sus soportes en el conjunto de
  prueba son de apenas 1–6 observaciones en varios cortes evaluados.
- Los datos cubren un periodo acotado (aprox. 2023–2024). No se
  capturan efectos estacionales de varios años (campañas, feriados
  recurrentes, inflación), que en un supermercado real sí inciden en
  la rotación y el riesgo de vencimiento.
- No hay variables exógenas (clima, promociones activas de la
  competencia, eventos locales) que en la práctica también explican
  variaciones bruscas de demanda.

## 2. Desbalance de clases y balanceo con SMOTE

- La clase BAJO concentra la gran mayoría de los lotes. Se incorporó
  SMOTE sobre el set de entrenamiento como mejora, pero SMOTE genera
  observaciones sintéticas por interpolación en el espacio de
  variables; con clases de 3–6 muestras reales, los vecinos
  sintéticos generados tienen poca diversidad y pueden no representar
  bien la variabilidad real de un futuro lote en riesgo CRÍTICO.
- Como resultado, la métrica de F1 por clase para CRITICO sigue
  siendo inestable entre distintos cortes de evaluación (ver
  `resultados/metricas_evaluacion.json` y el comparativo temporal en
  `evaluacion/evaluacion_modelos.py`). La exactitud global (>95%) es
  engañosa por sí sola y **no debe reportarse sin las métricas por
  clase** que ahora se generan junto a ella.

## 3. Modelo de reposición (`modelo_reposicion.py`)

- La regresión sobre ventas semanales usa únicamente rezagos de las
  3 semanas anteriores más variables de contexto del producto; no
  incorpora estacionalidad explícita (mes del año, día de pago,
  fin de semana) por falta de suficientes ciclos anuales completos
  en los datos disponibles.
- El margen de seguridad (15%) sobre la venta estimada es un
  parámetro fijo elegido por criterio de negocio razonable, no
  optimizado estadísticamente (por ejemplo, contra un nivel de
  servicio objetivo o el costo real de quiebre de stock vs.
  sobre-stock). Ese parámetro debería calibrarse por categoría en un
  trabajo futuro.
- Para productos con menos de `N_REZAGOS + MIN_SEMANAS_PARA_ML`
  semanas de historial, la recomendación cae a una regla simple
  (tasa de rotación anualizada / 365 × 7). Es una aproximación de
  arranque en frío razonable, pero no aprende de las particularidades
  del producto ni de la categoría más allá de ese promedio.
- No hay una integración explícita entre el modelo de riesgo y el de
  reposición: un producto puede recibir una sugerencia de pedido alta
  incluso si ya está en riesgo ALTO/CRÍTICO de vencimiento. En
  producción, ambos modelos deberían conciliarse (p. ej., limitar la
  cantidad sugerida cuando el mismo producto está en riesgo alto).

## 4. Chatbot / PLN (`chatbot/chatbot.py`)

- El clasificador de intenciones (TF-IDF + Regresión Logística) se
  entrena con un conjunto reducido de frases de ejemplo escritas a
  mano (~65 frases para 9 intenciones). Generaliza razonablemente a
  variaciones cercanas de esas frases, pero no a formulaciones muy
  distintas, jerga regional o errores ortográficos severos.
- La extracción de entidades (nombre de producto/categoría) usa
  coincidencia difusa por `difflib`, que puede confundir productos con
  nombres parecidos (p. ej. "Yogurt Laive" vs. "Leche Gloria" si el
  mensaje es ambiguo o muy corto).
- No hay manejo de contexto conversacional (el chatbot no recuerda de
  qué producto se habló en el mensaje anterior); cada mensaje se
  interpreta de forma aislada.

## 5. Evaluación y monitoreo en producción

- La separación de la evaluación en `evaluacion/evaluacion_modelos.py`
  agrega validación cruzada y comparación temporal, pero sigue siendo
  un análisis "offline": no existe un mecanismo de monitoreo continuo
  (data drift, degradación de métricas en producción) ni una política
  de reentrenamiento automático.
- El endpoint `/tendencia` es descriptivo (cuenta lotes históricos por
  nivel de riesgo y mes); no debe interpretarse como una proyección ni
  como evidencia causal de por qué cambia el riesgo mes a mes.

## 6. Infraestructura y despliegue

- El servidor Flask entrena ambos modelos en memoria cada vez que
  arranca (`app.py`), lo cual es adecuado para una demo académica pero
  no escalaría bien con un catálogo de miles de productos o múltiples
  tiendas; en ese escenario se recomendaría separar el entrenamiento
  (batch/offline, con modelos serializados) del servicio de
  predicción (online).
- No se implementó autenticación, control de acceso ni persistencia en
  base de datos (todo se lee de CSV en `data/`); para un piloto real en
  tienda se necesitaría una capa de datos transaccional.

## 7. Resumen de trabajo futuro sugerido

1. Ampliar el histórico de datos a varios años para capturar
   estacionalidad real y mejorar el soporte de las clases minoritarias.
2. Calibrar el margen de seguridad de reposición por categoría, usando
   costos reales de quiebre de stock vs. sobre-stock.
3. Conciliar las recomendaciones de riesgo y reposición en una sola
   capa de decisión.
4. Ampliar el dataset de entrenamiento del chatbot y evaluar su
   desempeño con un conjunto de prueba independiente (no solo frases
   de entrenamiento).
5. Incorporar monitoreo de drift y una política formal de
   reentrenamiento periódico.
6. Migrar de CSV a una base de datos transaccional si el proyecto pasa
   de piloto académico a producción real.
