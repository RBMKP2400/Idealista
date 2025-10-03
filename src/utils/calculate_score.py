import os
import pandas as pd
import json
from utils.loguru_conf import logger
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
import torch


class RealEstateScored():
    """
    Clase para calcular un score para cada propiedad normalizado por barrio.

    Atributos:
    df (pd.DataFrame): DataFrame con las propiedades y sus características.
    model (AutoModelForSequenceClassification): Modelo entrenado para predicción de reformado.
    tokenizer (AutoTokenizer): Tokenizador para la entrada del modelo.
    max_length (int): Longitud máxima de la entrada del modelo.

    device (str): Dispositivo en el que se encuentra el modelo (GPU o CPU).

    Métodos:
    predict_condition_reformed (df): Predice si una vivienda está reformada (1) o requiere reforma (0).
    calculate_score (df, by_column='Barrio'): Calcula un score para cada propiedad normalizado por barrio.
    """

    def __init__(self, df, model_dir):
        # Configuración tokenización
        self.max_length = 255
        
        # Revisar GPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.cuda.empty_cache() #Liberar cache de cuda

        # 1. Cargar el modelo entrenado
        self.load_best_model(model_dir)

        # 2. Obtener las predicciones y actualizar el DataFrame
        self.predict_condition_reformed(df)

        # 3. Calcular la puntuación y actualizar el DataFrame
        self.calculate_score()
        
    def load_best_model(self, model_dir="models/best_model"):
        """
        Carga el modelo entrenado en la ruta especificada.

        Args:
        model_dir (str): Ruta del directorio del modelo.

        Raises:
        ValueError: Si no se encuentra el directorio del mejor modelo.
        """
        if not os.path.exists(model_dir):
            raise ValueError(f"No se encontró el directorio del mejor modelo: {model_dir}")
        
        # Cargar el mejor modelo entrenado
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        logger.info("Fine-tuning completado. Mejor modelo cargado.")

    # ----------------------- Predicción -----------------------
    def predict_condition_reformed(self, df, target_column="Descripción"):
        """
        Predice si una vivienda está reformada (1) o requiere reforma (0).

        Si cualquier chunk de un registro obtiene 1, el registro completo es 1.
        Añade la columna 'Reformado' al DataFrame original.

        Args:
        df (pd.DataFrame): DataFrame con las propiedades y sus características.
        target_column (str): Columna con las descripciones de las viviendas.

        Returns:
        pd.DataFrame: DataFrame con la columna 'Reformado' agregada.
        """
        self.model.eval()
        predictions = []

        descriptions = df[target_column].tolist()

        for desc in tqdm(descriptions, desc="Prediciendo vivienda reformada:"):
            # Tokenizar con posibilidad de chunks
            tokenized = self.tokenizer(
                desc,
                truncation=True,
                max_length=self.max_length,
                return_overflowing_tokens=True,
                return_tensors="pt",
                padding="max_length"
            ).to(self.device)

            chunk_preds = []
            for i in range(tokenized["input_ids"].shape[0]):
                inputs = {
                    "input_ids": tokenized["input_ids"][i].unsqueeze(0),
                    "attention_mask": tokenized["attention_mask"][i].unsqueeze(0)
                }
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                    pred = torch.argmax(logits, dim=-1).item()
                chunk_preds.append(pred)

            # Si algún chunk es 1 → el registro completo es 1
            final_pred = 1 if any(p == 1 for p in chunk_preds) else 0
            predictions.append(final_pred)

        # Crear columna en el DataFrame original
        df["Reformado"] = predictions
        self.df = df.copy()
    
    def calculate_score(self, by_column = "Barrio"):
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
        
        # Asegurar tipos correctos
        self.df[numeric_cols] = self.df[numeric_cols].apply(pd.to_numeric, errors='coerce').fillna(0).astype(float)
        self.df[binary_cols] = self.df[binary_cols].apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)
                
        # Copia para no modificar el DataFrame original
        df_norm = self.df[[id_col, by_column] + numeric_cols + binary_cols].copy()
        
        # Calcular min y max por barrio
        min_vals = df_norm.groupby(by_column)[numeric_cols + binary_cols].transform("min")
        max_vals = df_norm.groupby(by_column)[numeric_cols + binary_cols].transform("max")
        
        # Normalización min-max por barrio
        df_norm[numeric_cols + binary_cols] = (df_norm[numeric_cols + binary_cols] - min_vals) / (max_vals - min_vals)
        # Invertir "Precio por metro cuadrado" para que valores bajos sean mejores
        df_norm["Precio por metro cuadrado"] = 1 - df_norm["Precio por metro cuadrado"]
        
        # Evitar división por cero (si max == min)
        df_norm[numeric_cols + binary_cols] = df_norm[numeric_cols + binary_cols].fillna(0)
        
        # Calcular score ponderado
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
        
        # Calcular score
        df_norm["Score"] = sum(df_norm[col] * w * 10 for col, w in weights.items()).round(1)
        
        # Guardar el score en el DataFrame original
        self.df["Score"] = df_norm["Score"]
        

# ----------------------- Uso -----------------------
if __name__ == "__main__":
    # Si ya has entrenado previamente y quieres usar el mejor modelo:
    with open("output/idealista_registros.json", 'r', encoding='utf-8') as f:
        data = json.load(f)
    df = pd.DataFrame(data)

    scorer = RealEstateScored(df, model_dir="models/best_model")

    df = scorer.df.copy()
