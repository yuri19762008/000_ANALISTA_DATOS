# Conclusión del análisis de reducción de dimensionalidad en el dataset *digits*

A continuación presento una conclusión estructurada paso a paso, comparando **PCA**, **t‑SNE** (con dos perplejidades) y el **autoencoder** aplicado al dataset `digits` de `sklearn`.

---

## 1. Recordatorio del objetivo del ejercicio

1. El dataset `digits` contiene imágenes de dígitos escritos a mano representadas como vectores de 64 características (8x8 píxeles aplanados).  
2. El objetivo principal fue **reducir la dimensionalidad a 2 componentes** para poder visualizar los datos en un plano y analizar si existen **agrupamientos (clusters)**, patrones o anomalías.  
3. Para ello se aplicaron y compararon tres técnicas distintas:
   - PCA (método lineal).  
   - t‑SNE (método no lineal, probado con dos perplejidades).  
   - Autoencoder (modelo de red neuronal que aprende una representación 2D).

---

## 2. Observaciones sobre PCA (2 componentes)

1. **Estructura global de los datos**  
   - PCA proyecta los datos en dos componentes principales que maximizan la varianza explicada.  
   - En el scatter plot se aprecia cierta separación entre varios dígitos, pero también **zonas de solapamiento** donde diferentes clases comparten la misma región del plano.

2. **Claridad de los clusters**  
   - Algunos dígitos aparecen algo más compactos y separados (por ejemplo, ciertos dígitos con trazos muy característicos), pero en general los clusters no son perfectamente definidos.  
   - Esto es consistente con la naturaleza lineal de PCA: al usar solo combinaciones lineales de las 64 variables, le cuesta capturar separaciones que dependan de relaciones no lineales entre píxeles.

3. **Estabilidad y parametrización**  
   - PCA es muy **estable**: cambiando poco el código o repitiendo la ejecución, la figura se mantiene prácticamente igual.  
   - Tiene pocos parámetros (básicamente el número de componentes), por lo que es fácil de usar e interpretar.  
   - Además, es rápido computacionalmente incluso para datasets grandes.

**Conclusión parcial sobre PCA:**  
PCA ofrece una visión **global y estable** de la estructura de los datos y permite detectar tendencias generales y cierta separación entre clases, pero no logra separar completamente todos los dígitos en clusters bien definidos.

---

## 3. Observaciones sobre t‑SNE (perplejidad 30 y 50)

### 3.1. t‑SNE con perplejidad 30

1. **Estructura local y clusters**  
   - Con perplejidad 30, t‑SNE genera típicamente **clusters bastante bien definidos**, donde cada dígito se agrupa en nubes de puntos relativamente compactas.  
   - Muchos dígitos se separan visualmente mejor que en PCA, con menos solapamiento entre clases.

2. **Preservación de vecindarios**  
   - t‑SNE se enfoca en preservar las relaciones de **vecinos cercanos**, por lo que puntos que eran similares en 64 dimensiones siguen cerca en el plano 2D.  
   - Esto se refleja en agrupamientos que tienen sentido semántico (ejemplo: dígitos con formas parecidas quedando cerca entre sí).

### 3.2. t‑SNE con perplejidad 50

1. **Cambio en la forma de los clusters**  
   - Al aumentar la perplejidad a 50, la estructura de los clusters cambia: algunos grupos se vuelven más grandes o se fusionan, otros se reacomodan.  
   - De todos modos, se mantiene una separación razonable entre muchas clases, aunque no siempre tan marcada como con perplejidad 30.

2. **Sensibilidad a hiperparámetros**  
   - El hecho de que el mapa cambie de manera apreciable solo por modificar la perplejidad muestra que t‑SNE es **altamente sensible a sus parámetros** y a la inicialización.  
   - Esto implica que, para usarlo correctamente, es necesario experimentar con varios valores de perplejidad y revisar varias ejecuciones.

**Conclusión parcial sobre t‑SNE:**  
t‑SNE produce las visualizaciones con **clusters más claros y separados** entre dígitos, especialmente con perplejidad 30, lo que lo hace muy útil para **exploración visual de agrupamientos**. Sin embargo, su alta sensibilidad a parámetros y a la aleatoriedad lo hace menos estable y más difícil de reproducir que PCA.

---

## 4. Observaciones sobre el autoencoder (espacio latente 2D)

1. **Representación no lineal aprendida**  
   - El autoencoder utiliza capas densas con activación ReLU para comprimir los 64 píxeles originales en un espacio latente de solo 2 dimensiones y luego reconstruir la entrada.  
   - Al entrenarse para minimizar el error de reconstrucción, el modelo aprende una **representación 2D no lineal** que captura patrones relevantes de los dígitos.

2. **Distribución de los puntos en el espacio latente**  
   - En el scatter plot del espacio latente se observan zonas donde varios dígitos forman grupos relativamente compactos, y otras donde hay mayor mezcla de clases.  
   - Dependiendo de la inicialización y del entrenamiento, la forma exacta de los clusters puede cambiar, pero en general se aprecia cierta estructura donde algunos dígitos quedan más próximos a otros con formas similares.

3. **Flexibilidad y sensibilidad a la configuración**  
   - El autoencoder es **muy flexible**: cambiando el número de neuronas, la profundidad o las funciones de activación se pueden obtener representaciones muy diferentes.  
   - Esta flexibilidad implica también **sensibilidad** a la arquitectura, a la tasa de aprendizaje, al número de épocas y a la aleatoriedad propia del entrenamiento de redes neuronales.  
   - A diferencia de PCA, no es tan fácil interpretar qué significa cada dimensión latente en términos de las variables originales.

**Conclusión parcial sobre el autoencoder:**  
El autoencoder ofrece una representación 2D no lineal que puede capturar relaciones complejas entre píxeles y producir clusters razonables, pero su comportamiento depende mucho del diseño del modelo y del entrenamiento, por lo que requiere más experiencia para ajustarlo y no es tan interpretable como PCA.

---

## 5. Comparación global entre las tres técnicas

1. **Claridad de agrupamientos**
   - PCA: proporciona clusters **moderadamente separados**, con solapamiento notable entre algunas clases.  
   - t‑SNE (perplejidad 30/50): logra los **clusters más definidos y visualmente separados**, especialmente para valores de perplejidad intermedios.  
   - Autoencoder: genera una estructura intermedia, con ciertos clusters claros y otras regiones más mezcladas, dependiendo del entrenamiento.

2. **Sensibilidad a los parámetros y a la aleatoriedad**
   - PCA: muy **poco sensible**, resultados estables y reproducibles.  
   - t‑SNE: **muy sensible** a la perplejidad, la semilla y otros parámetros, los mapas pueden cambiar bastante.  
   - Autoencoder: sensible a la arquitectura, al optimizador y al número de épocas; requiere fijar semillas y configurar bien el modelo para tener resultados consistentes.

3. **Interpretabilidad**
   - PCA: las componentes son combinaciones lineales de las variables originales; se puede analizar cuánta varianza explica cada componente.  
   - t‑SNE: difícil de interpretar cuantitativamente; se usa sobre todo como herramienta visual para ver vecindarios y clusters.  
   - Autoencoder: las dimensiones latentes son aún menos interpretables directamente, aunque pueden capturar factores de variación interesantes.

4. **Coste computacional**
   - PCA: muy rápido y eficiente incluso con muchos datos.  
   - t‑SNE: más costoso; por eso es habitual reducir primero con PCA (por ejemplo a 30 dimensiones) antes de aplicar t‑SNE.  
   - Autoencoder: el tiempo depende del tamaño de la red y del número de épocas; suele ser más costoso que PCA y comparable o superior a t‑SNE.

---

## 6. Elección de la técnica preferida y justificación

1. **Para exploración visual de clusters en este dataset concreto (`digits`):**  
   - La técnica que ofrece **la separación visual más clara entre clases** es t‑SNE con una perplejidad intermedia (por ejemplo, 30), donde los dígitos aparecen organizados en nubes compactas y bien diferenciadas.  
   - Esto facilita mucho la identificación de grupos y posibles confusiones entre dígitos que quedan cerca en el mapa.

2. **Para un análisis rápido, estable e interpretable:**  
   - PCA es la opción más recomendable, ya que es **fácil de ejecutar, reproducible** y permite explicar cuánta varianza de los datos se concentra en las componentes principales.  
   - Aunque no separa perfectamente todos los dígitos, ofrece una visión global robusta con muy poco ajuste de parámetros.

3. **Para aprender representaciones no lineales reutilizables en otros modelos:**  
   - El autoencoder resulta interesante porque la capa latente de 2 dimensiones (o más, si se quisiera ampliar) puede usarse como **features comprimidas** para otras tareas de clasificación o visualización.  
   - Sin embargo, requiere más tiempo de configuración y entrenamiento, y su interpretación es menos directa.

**Conclusión final:**  
- Si el objetivo principal es **exploración visual de agrupamientos** y se acepta trabajar con un método más sensible a parámetros, la técnica más útil para este dataset es **t‑SNE**, ya que logra los clusters de dígitos más claros y separados.  
- Si se prioriza la **estabilidad, rapidez e interpretabilidad**, **PCA** es la opción más adecuada, aun cuando los clusters no sean tan perfectos.  
- El **autoencoder** se posiciona como una alternativa flexible para aprender representaciones no lineales reutilizables, pero requiere más experiencia y cuidado en su configuración, por lo que lo considero un complemento interesante más que la primera opción para visualización exploratoria en este caso.

---
