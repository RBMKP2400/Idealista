from connection.idealista_conn import IdealistaConnection
from connection.gmail_conn import GmailSender
from params.idealista_params import get_params
from utils.idealista_utils import parse_response, filter_output, table_metics
from utils.help import get_config, save_df_to_json_append
from utils.loguru_conf import logger
import time
import pandas as pd
import os

# Carga la configuración del proyecto desde un archivo YAML
path_config = os.getenv('CONFIG_PATH', '.\\src\\config.yaml')
config = get_config(path_config)
if not config:
    logger.error("No se pudo cargar la configuración del proyecto.")
    exit(1)
# Inicializa la conexión con Idealista
idealista_conn = IdealistaConnection()

# Extrae y almacena los datos
all_new_records = []
for shape_type in ['Mid Home']: #['Small Home', 'Mid Home', 'Big Home']
    for location in ["La Latina", "Alcorcón", "Leganés", "Carabanchel"]:
        logger.info(f"Procesando {shape_type} en {location}...")
        # Obtiene los parámetros
        params = get_params(shape_type, location)
        # Realiza la solicitud
        status, response = idealista_conn.api_post(params)
        # Parsea la respuesta
        df = parse_response(status, response)
        # Aplica filtros adicionales al DataFrame
        df = filter_output(df)
        # Guarda el DataFrame en un archivo JSON
        new_records = save_df_to_json_append(df, config['PATH_RESULTS'])
        all_new_records.extend(new_records)
        time.sleep(1)

# Crea la tabla resumen de los datos
df_metrics = table_metics(config)
df_new_records = pd.DataFrame(all_new_records)

# GMAIL: Enviar correo
if df_new_records.empty:
    logger.info("No se encontraron registros nuevos. No se envía correo.")
else:
    # Eliminar columnas innecesarias
    df_new_records.drop(columns=["Notas", "Descripción","Número de fotos", "Tiene vídeo", "Tiene plano"], inplace=True)

    # Establecer las configuraciones del correo
    sender = GmailSender(config)

    # Guardar el Excel
    excel_path = config['GMAIL_EXCEL']
    df_new_records.to_excel(excel_path, index=False)

    # Enviar mensaje
    sender.send_message(df_new_records, df_metrics, excel_path)

logger.info("Hasta la próxima!")