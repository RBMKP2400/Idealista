# Tu Asistente de Búsqueda de Viviendas

Este proyecto permite consultar la API de Idealista para obtener listados de viviendas en venta y enviar un correo electrónico automático con los resultados filtrados.  
Los datos se guardan en JSON y Excel para un fácil análisis posterior.  




## Características principales

- Autenticación automática con la API de Idealista (gestiona el Bearer Token y su expiración).  
- Configuración para diferentes tipos de vivienda en función de su tamaño (`params/idealista_params.py`):  
  - Big Home  
  - Mid Home  
  - Small Home  
  Además, se realiza un **segundo filtrado** de los resultados mediante la función `filter_output`, que permite refinar aún más los registros obtenidos según criterios personalizados.  
  Se recomienda que adaptes esta función a tus necesidades específicas para obtener solo los registros que realmente te interesen.  
- Persistencia de resultados en `output/` en formato JSON y Excel.  
- Historial en memoria de todos los registros extraídos de la API.  
- Notificaciones por correo electrónico:  
  - Envía solo nuevos registros encontrados.  
  - Incluye un Excel con todos los registros históricos.  
  - El cuerpo del correo muestra una tabla con los registros nuevos.  
  - Si no hay registros nuevos, no se envía correo.  
- Preparado para ejecutarse en Docker con volúmenes persistentes (`/output`, `/tokens`). 



## Ubicaciones preseleccionadas

El proyecto realiza búsquedas en las siguientes zonas de Madrid:

- Latina  
- Carabanchel  
- Alcorcón  
- Leganés  


## Configuración

### 1. Variables de entorno (`.env`)

Copia el archivo de ejemplo y complétalo con tus credenciales.

### 2. Archivo de configuración (config.yaml)

Copia el archivo de ejemplo y complétalo según tus necesidades.
