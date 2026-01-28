# Proyecto: Preparación de Datos con Python

## 1. Introducción

Este proyecto implementa un flujo completo de **obtención, limpieza, transformación y estructuración de datos** utilizando Python junto con las librerías **NumPy** y **Pandas**. El objetivo final es generar un dataset limpio y consolidado, listo para ser utilizado en análisis posteriores y modelos de machine learning. [file:13]

Los datos principales corresponden a clientes de e‑commerce, complementados con información ficticia generada en NumPy y con fuentes externas en formatos CSV, Excel y una tabla web. [file:13][file:10][file:11]

---

## 2. Justificación del uso de NumPy y Pandas

- **NumPy** se utilizó para generar datos numéricos ficticios (IDs, edades, número de compras, montos) de forma vectorizada y eficiente, permitiendo aplicar operaciones como medias, sumas y conteos sobre arrays sin necesidad de bucles explícitos. [file:13]
- **Pandas** se empleó para convertir estos datos en DataFrames, explorar y transformar la información, combinar múltiples fuentes (CSV, Excel, web) y aplicar técnicas de limpieza, data wrangling, agrupamiento y exportación a formatos finales (CSV y Excel). [file:13]

La combinación de NumPy + Pandas permite un flujo de trabajo compacto, legible y escalable para la preparación de datos en proyectos reales de analítica. [file:13]

---

## 3. Descripción de los datos y fuentes utilizadas

### 3.1 Datos generados con NumPy (Lección 1)

Se generó un conjunto de datos ficticio de clientes con: [file:13]

- ID de cliente  
- Edad  
- Total de compras realizadas  
- Monto total aproximado gastado  

Estos datos se almacenaron en un array NumPy y luego en el archivo `clientes_numpy.npy`, que sirve como insumo para la carga en Pandas. [file:13]

### 3.2 Datos en Pandas desde NumPy (Lección 2)

Los datos de NumPy se cargaron en un **DataFrame de Pandas**, donde se realizó: [file:13]

- Visualización de primeras y últimas filas.  
- Cálculo de estadísticas descriptivas.  
- Filtros condicionales (por edad, número de compras, etc.).  

El resultado preliminar se exportó a `clientes_desde_numpy.csv` para ser reutilizado en la siguiente etapa. [file:13]

### 3.3 Fuentes externas (Lección 3)

Además del CSV generado, se integraron: [file:13][file:10][file:11]

- Archivo **`clientes_ecommerce.csv`** con información realista de clientes (ID, Nombre, Edad, Ciudad, Total_Compras, Monto_Total).  
- Archivo **`clientes_ecommerce.xlsx`** con la misma estructura de clientes en formato Excel.  
- Una **tabla web** obtenida mediante `read_html()` (vía `requests` + `lxml`), utilizada como ejemplo de incorporación de datos desde páginas web.  

Estos datos se unificaron con el DataFrame base mediante operaciones de merge y concatenación, generando un **DataFrame consolidado** (`dataset_consolidado.csv`). [file:13]

---

## 4. Técnicas de limpieza y transformación aplicadas

### 4.1 Manejo de valores perdidos y outliers (Lección 4)

A partir del DataFrame consolidado se realizaron las siguientes tareas: [file:13]

- Identificación de **valores nulos** por columna.  
- Imputación de valores faltantes en la columna de Edad utilizando la media.  
- Detección de **outliers** en la variable `Monto_Total` mediante el método IQR (cuartil 1, cuartil 3 y rango intercuartílico).  
- Tratamiento de outliers recortando los valores fuera de los límites definidos.  

El resultado se guardó como `dataset_limpio.csv`, sirviendo de base para el data wrangling. [file:13]

### 4.2 Data Wrangling (Lección 5)

Sobre el DataFrame limpio se aplicaron técnicas de **manipulación avanzada**: [file:13]

- Eliminación de registros duplicados (previa simulación de duplicados para ejemplificar el proceso).  
- Transformación de tipos de datos, por ejemplo, conversión de la columna `Ciudad` a tipo categórico.  
- Creación de nuevas columnas calculadas:
  - `Ticket_Promedio`: monto total dividido por el número de compras.  
  - `Segmento_Monto`: clasificación del monto total en rangos (Bajo, Medio, Alto).  
  - `Monto_Normalizado`: versión normalizada de `Monto_Total` usando escalado min‑max.  
- Uso de funciones personalizadas con `apply`, `lambda` y otras herramientas de Pandas para enriquecer la estructura del dataset.  

El DataFrame resultante se almacenó en `dataset_wrangle.csv`. [file:13]

### 4.3 Agrupamiento y pivoteo de datos (Lección 6)

En la última etapa se organizó la información para facilitar el análisis: [file:13]

- Agrupamiento por **Ciudad** para obtener métricas agregadas (suma de compras, suma y media de montos, ticket promedio).  
- Construcción de tablas **pivot** cruzando `Ciudad` con `Segmento_Monto`, utilizando `pivot_table` para obtener montos promedio por segmento.  
- Uso de `melt` para reestructurar tablas pivotadas a un formato largo, adecuado para ciertos tipos de análisis y visualización.  
- Exportación del **dataset final estructurado** en los archivos `dataset_final.csv` y `dataset_final.xlsx`. [file:13]

---

## 5. Principales decisiones y desafíos

Durante el desarrollo del flujo se tomaron varias decisiones técnicas: [file:13]

- Elegir la media como método de imputación de la Edad faltante para mantener un valor numérico coherente con el conjunto.  
- Tratar los outliers de `Monto_Total` con recorte (clipping) en lugar de eliminar registros, con el fin de conservar la mayor parte posible de la información.  
- Convertir campos como `Ciudad` a tipo categórico para mejorar la legibilidad y potencialmente la eficiencia en operaciones de agrupamiento.  
- Manejar dependencias externas (`lxml`, `requests`) y posibles errores HTTP (como códigos 403 en la lectura web), documentando la lectura desde la web como parte opcional pero incluida en el flujo. [file:13]

Estos desafíos y decisiones forman parte importante del proceso de preparación de datos en contextos reales, donde las fuentes presentan formatos heterogéneos y problemas de calidad. [file:13]

---

## 6. Resultados y estado final del dataset

El resultado final del proyecto es un **dataset consolidado, limpio y enriquecido**, con: [file:13][file:10][file:11]

- Información integrada de clientes desde NumPy, CSV, Excel y web.  
- Valores nulos tratados y outliers controlados.  
- Nuevas variables derivadas que facilitan el análisis (segmentos, tickets promedio, normalizaciones).  
- Estructuración adecuada para aplicar agrupamientos, pivoteos y visualizaciones, tanto en notebooks como en un dashboard adicional (por ejemplo, usando Streamlit).  

Este dataset final, entregado en formatos **CSV y Excel**, está listo para ser utilizado en reportes de negocio y modelos predictivos, cumpliendo con los requerimientos del proyecto de “Preparación de Datos con Python”. [file:13]
