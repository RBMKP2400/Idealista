import pandas as pd
import json
from datetime import datetime
import os
import yaml
from pathlib import Path
from utils.loguru_conf import logger


def get_new_records(df: pd.DataFrame, df_new: pd.DataFrame, location: str) -> pd.DataFrame:
    """
    Devuelve los registros de df_new que no están presentes en df
    basándose en la combinación de 'Property Code' y 'Price'.

    Args:
        df (pd.DataFrame): DataFrame existente con los registros actuales.
        df_new (pd.DataFrame): DataFrame nuevo con posibles registros nuevos.

    Retorna:
        pd.DataFrame: DataFrame con los registros de df_new que no se encuentran en df.
    """

    # Asegurarse de que no haya problemas con espacios o formatos en las columnas clave
    for col in ["Código de propiedad", "Precio"]:
        df[col] = df[col].astype(str).str.strip()
        df_new[col] = df_new[col].astype(str).str.strip()

    # Crear una columna clave combinada para identificar registros únicos
    df["_key"] = df["Código de propiedad"] + "_" + df["Precio"]
    df_new["_key"] = df_new["Código de propiedad"] + "_" + df_new["Precio"]

    # Filtrar sólo los registros nuevos que no existen en df
    new_records = df_new[~df_new["_key"].isin(df["_key"])].copy()
    expired_records = df[(~df["_key"].isin(df_new["_key"])) & (df["Municipio"] == location)].copy()

    # Convertir 'Precio' a numérico, rellenar NaNs con 0 y convertir a entero
    new_records["Precio"] = pd.to_numeric(new_records["Precio"], errors="coerce").fillna(0).astype(int)
    expired_records["Precio"] = pd.to_numeric(expired_records["Precio"], errors="coerce").fillna(0).astype(int)

    # Eliminar la columna clave temporal en ambos DataFrames
    new_records.drop(columns="_key", inplace=True)
    expired_records.drop(columns="_key", inplace=True)
    df.drop(columns="_key", inplace=True)

    return new_records, expired_records


def save_df_to_json_append(df: pd.DataFrame, filename: str, location: str):
    """
    Guarda un DataFrame en un archivo JSON, añadiendo registros si el archivo ya existe.
    Crea automáticamente el directorio si no existe.

    Args:
        df (pd.DataFrame): El DataFrame que se quiere guardar.
        filename (str): La ruta del archivo JSON.

    Retorna:
        json: Una lista de diccionarios con los nuevos registros añadidos al archivo.
    """
    filename = Path(filename)
    # Crear el directorio padre si no existe
    filename.parent.mkdir(parents=True, exist_ok=True)

    # Crear una copia del DataFrame y añadir una columna 'datetime' con la fecha y hora actual
    df = df.copy()
    df['datetime'] = datetime.now().isoformat()
    df['Precio'] = pd.to_numeric(df["Precio"], errors="coerce").fillna(0).astype(int)

    # Inicializar la lista de datos leyendo del archivo si existe
    if filename.exists():
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Asegurarse que data es una lista
                if not isinstance(data, list):
                    data = []
        except json.JSONDecodeError:
            data = []
    else:
        data = []

    # Determinar los nuevos registros para añadir
    if data:
        # Obtener nuevos registros comparando los datos existentes con el nuevo DataFrame
        new_records, expired_records = get_new_records(pd.DataFrame(data), df, location)
        new_records = new_records.to_dict(orient='records')

        if not expired_records.empty:
            # Obtener los códigos de propiedad a eliminar
            drop_records = set(expired_records["Código de propiedad"].unique())
            # Filtrar data eliminando los registros viejos inexistentes
            data = [rec for rec in data if rec["Código de propiedad"] not in drop_records]
        else:
            drop_records = []
    else:
        new_records = df.to_dict(orient='records')
        drop_records = []

    # Construir/actualizar un diccionario indexado por Código de propiedad
    store = {rec["Código de propiedad"]: rec for rec in data}

    if new_records:
        for rec in new_records:
            codigo = rec["Código de propiedad"]
            fecha = rec["datetime"]
            precio = rec["Precio"]

            if codigo not in store:
                # Si es un Registro nuevo se inicialia histórico
                rec["Histórico de precios"] = {fecha: precio}
                store[codigo] = rec
            else:
                # Actualizar histórico
                if "Histórico de precios" not in store[codigo]:
                    store[codigo]["Histórico de precios"] = {}
                store[codigo]["Histórico de precios"][fecha] = precio

                # Si es más reciente,se actulizan todos los campos
                if pd.to_datetime(fecha) > pd.to_datetime(store[codigo]["datetime"]):
                    store[codigo].update(rec)
    else:
        for rec in data:
            codigo = rec["Código de propiedad"]
            store[codigo].update(rec)

    # Actualizar el archivo JSON con los nuevos registros
    if store:
        # Escribir los datos actualizados en el archivo
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(list(store.values()), f, ensure_ascii=False, indent=4)

    # Mostrar el número de nuevos registros añadidos
    logger.info(f"{len(new_records)} registros añadidos al archivo {filename}.")
    logger.info(f"{len(drop_records)} registros expirados se eliminan del archivo {filename}.")

    return new_records


def get_config(path_config: str = 'config.yaml') -> dict:
    """
    Carga la configuración del proyecto desde un archivo YAML.

    Lee el archivo YAML especificado y devuelve su contenido como un diccionario.
    Si el archivo no existe o no se puede leer, devuelve un diccionario vacío.

    Args:
        path_config (str): Ruta del archivo de configuración (por defecto 'src/config.yaml').

    Returns:
        dict: Contenido del archivo de configuración.
    """
    try:
        # Abrimos el archivo YAML en modo lectura
        with open(os.path.normpath(path_config), 'r', encoding='utf-8') as f:
            # Cargamos el contenido del archivo en un diccionario
            config = yaml.safe_load(f)
        # Devolvemos el diccionario
        return config
    except FileNotFoundError:
        # Si el archivo no existe, imprimimos un mensaje de error y devolvemos un diccionario vacío
        logger.error(f"Error: No se encontró el archivo de configuración en {path_config}")
        return {}
    except yaml.YAMLError as e:
        # Si hay un error al parsear YAML, imprimimos un mensaje de error y devolvemos un diccionario vacío
        logger.error(f"Error al parsear YAML: {e}")
        return {}
    
def json_a_excel(json_path: str, excel_path: str) -> None:
    """
    Convierte un archivo JSON en un archivo Excel.

    Args:
        json_path (str): Ruta del archivo JSON.
        excel_path (str): Ruta del archivo Excel que se va a crear.
    """
    # Carga el archivo JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Convierte el JSON en un DataFrame
    df = pd.json_normalize(data)

    # Guarda el DataFrame en un archivo Excel
    df.to_excel(excel_path, index=False)

    logger.info(f"Excel generado en: {excel_path}")
