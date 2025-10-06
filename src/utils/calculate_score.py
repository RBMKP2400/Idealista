import pandas as pd

def calculate_score(df, by_column = "Barrio"):
    """
    Calcula un score para cada propiedad normalizado por barrio.

    El score se calcula como una ponderación de las características de la propiedad.
    Se utiliza una normalización min-max por barrio para que los valores estén entre 0 y 1.
    Luego se calcula la ponderación de cada característica y se suma para obtener el score total.

    Devuelve un DataFrame con cada registro identificado por 'Código de propiedad' 
    y su score individual.

    Args:
    by_column (str): Columna por la que se normaliza el score (por defecto "Barrio").

    Returns:
    pd.DataFrame: DataFrame con cada registro identificado por 'Código de propiedad' 
    y su score individual.
    """
    # Columnas
    id_col = "Código de propiedad"
    numeric_cols = ["Precio por metro cuadrado", "Habitaciones", "Baños"]
    binary_cols = ["Nueva promoción", "Exterior", "Tiene ascensor", "Plaza de parking", "Reformado"]

    # Pesos ponderados
    weights = {
        "Precio por metro cuadrado": 0.4, #ponderación por metro cuadrado
        "Habitaciones": 0.2, #ponderación por habitaciones
        "Baños": 0.1, #ponderación por baños
        "Nueva promoción": 0.1, #bonus 10%
        "Reformado": 0.1, #bonus 10%
        "Tiene ascensor": 0.15, #ponderación por ascensor
        "Exterior": 0.05, #ponderación por exterior
        "Plaza de parking": 0.1, #ponderación por plaza de parking
    }
    
    # Asegurar tipos correctos
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce').fillna(0).astype(float)
    df[binary_cols] = df[binary_cols].apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)
            
    # Copia para no modificar el DataFrame original
    df_norm = df[[id_col, by_column] + numeric_cols + binary_cols].copy()
    
    # Calcular min y max por barrio
    min_vals = df_norm.groupby(by_column)[numeric_cols + binary_cols].transform("min")
    max_vals = df_norm.groupby(by_column)[numeric_cols + binary_cols].transform("max")
    
    # Normalización min-max por barrio
    df_norm[numeric_cols + binary_cols] = (df_norm[numeric_cols + binary_cols] - min_vals) / (max_vals - min_vals)
    # Invertir "Precio por metro cuadrado" para que valores bajos sean mejores
    df_norm["Precio por metro cuadrado"] = 1 - df_norm["Precio por metro cuadrado"]
    
    # Evitar división por cero (si max == min)
    df_norm[numeric_cols + binary_cols] = df_norm[numeric_cols + binary_cols].fillna(0)
    
    # Calcular score
    df_norm["Score"] = sum(df_norm[col] * w * 10 for col, w in weights.items()).round(1)
    
    # Guardar el score en el DataFrame original
    df["Score"] = df_norm["Score"]

    return df