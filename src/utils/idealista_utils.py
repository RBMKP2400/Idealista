import pandas as pd
import re
import numpy as np
import json
from utils.loguru_conf import logger

def parse_response(status: int, response: dict) -> pd.DataFrame:
    """
    Parsea la respuesta de Idealista y devuelve un DataFrame con los datos.

    Args:
        status (int): Código de estado HTTP de la respuesta.
        response (dict): Diccionario con la respuesta de la API de Idealista.

    Returns:
        pd.DataFrame: Un DataFrame con los datos de la respuesta.
    """
    atributes = {
        "propertyCode": "Código de propiedad",
        "municipality": "Municipio",
        "district": "Distrito",
        "neighborhood": "Barrio",
        "address": "Dirección",
        "status": "Estado",
        "floor": "Planta",
        "price": "Precio",
        "priceByArea": "Precio por metro cuadrado",
        "propertyType": "Tipo de propiedad",
        "newDevelopment": "Nueva promoción",
        "size": "Superficie (m²)",
        "exterior": "Exterior",
        "rooms": "Habitaciones",
        "bathrooms": "Baños",
        "hasLift": "Tiene ascensor",
        "parkingSpace": "Plaza de parking",
        "url": "URL",
        "description": "Descripción",
        "numPhotos": "Número de fotos",
        "hasVideo": "Tiene vídeo",
        "hasPlan": "Tiene plano",
        "notes": "Notas"
    }

    # Verifica si la respuesta fue exitosa
    if status != 200:
        logger.warning(f"Aviso: La consulta no fue exitosa: {status}")
        return pd.DataFrame()  # Devuelve un df vacío en caso de error

    # Parsea la respuesta
    df = pd.DataFrame(response.get("elementList", []))

    # Solo procesa si hay datos
    if df.empty:
        logger.warning("Aviso: No hay datos en la respuesta.")
        return df

    # Selecciona columnas y renombra
    columnas = [col for col in atributes.keys() if col in df.columns]
    df = df[columnas].rename(columns={k:v for k,v in atributes.items() if k in columnas})

    # Nueva columna
    df["Rango Superficie"] = np.where(df["Superficie (m²)"] < 90, "<90 m²", ">90 m²")

    return df

def filter_output(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filtra los datos de un DataFrame de acuerdo a criterios fijos:
      - Excluye 'nuda' en descripción
      - Excluye variantes de 'okupa'
      - Excluye 'usera' en descripción
      - Excluye planta 'bj'
      - Filtra por lista de barrios relevantes
    """

    # Limpiar espacios y poner en minúsculas para todas las columnas relevantes
    for col in ["Descripción", "Planta", "Barrio"]:
        df[col] = df[col].astype(str).str.strip()

    # Filtros por 'nuda', 'okupa' y 'usera'
    df = df[~df["Descripción"].str.lower().str.contains("nuda")]
    df = df[~df["Descripción"].str.lower().str.contains(r"okupa\w*", flags=re.IGNORECASE)]
    df = df[~df["Descripción"].str.lower().str.contains(r"ocupa\w*", flags=re.IGNORECASE)]
    df = df[~df["Descripción"].str.lower().str.contains("usera")]
    df = df[~df["Descripción"].str.lower().str.contains("olivos")]
    df = df[~df["Descripción"].str.lower().str.contains("pan bendito")]
    df = df[~df["Descripción"].str.lower().str.contains(r"alquil\w*", flags=re.IGNORECASE)]

    # Filtro por dirección
    df[~(df["Dirección"].str.lower().str.contains("carlos heredero")) & (df["Barrio"].str.lower().str.contains("puerta bonita"))] # cerca de abrantes
    df[~(df["Dirección"].str.lower().str.contains("belzunegui")) & (df["Barrio"].str.lower().str.contains("puerta bonita"))] # cerca de abrantes
    df[~(df["Dirección"].str.lower().str.contains(r"avefr[ií]a")) & (df["Barrio"].str.lower().str.contains("puerta bonita"))] # cerca de abrantes

    # Filtro por planta
    df = df[~(df["Planta"].str.lower().isin(["ss", "bj"]))]

    # Lista de barrios no válidos
    barrios_no_validos = [
        "los cármenes",
        "abrantes",
        "san nicasio", "zarzaquemada", "el carrascal", "los frailes"
    ]
    
    # Filtro por barrio
    df = df[~(df["Barrio"].str.lower().isin(barrios_no_validos))]

    # Filtro ordenar por precio
    df = df.sort_values(by=["Municipio", "Distrito", "Barrio", "Rango Superficie", "Precio"], ascending=True).reset_index(drop=True)

    return df

def table_metics(config):
    """
    Reads property data from a JSON file, processes it, and returns aggregated metrics.

    Args:
        config (dict): Configuration dictionary containing the path to results.

    Returns:
        pd.DataFrame: A DataFrame with aggregated metrics by municipality, district, neighborhood, and surface range.
    """
    # Attempt to open and read the JSON file
    try:
        with open(config["PATH_RESULTS"], 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        logger.error(f"Error al leer el archivo JSON: {config["PATH_RESULTS"]}")
        return pd.DataFrame()  # Return an empty DataFrame if JSON decoding fails

    # Check if data was loaded successfully
    if data:
        # Convert the data into a DataFrame
        df = pd.DataFrame(data)

        # Group data by specific columns and calculate mean price and count
        result = (
            df.groupby(["Municipio", "Distrito", "Barrio", "Rango Superficie"])
            .agg(
                Precio_medio=("Precio", "mean"),
                Num_registros=("Precio", "count")
            )
            .reset_index()
        )

        # Convert the mean price to an integer
        result["Precio_medio"] = result["Precio_medio"].astype(int)
        return result

    # Return an empty DataFrame if no data is present
    return pd.DataFrame()
