import os
import pandas as pd
from utils.loguru_conf import logger
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
import spacy
from sentence_transformers import SentenceTransformer, util
import torch
from datasets import Dataset, DatasetDict, load_from_disk




class TrainClassificationLLM:
    """
    Recomendador de inmuebles usando LLM grande con fine-tuning,
    pseudo-labeling profesional (lematización + embeddings) y early stopping.

    1. Pseudo-labeling profesional:
        - Lematización en español para capturar lexemas.
        # python -m spacy download es_core_news_sm
        - Embeddings para similitud semántica (listo para entrar vs requiere reforma).

    2. Fine-tuning robusto:
        - Early stopping automático basado en eval_loss.
        - Logging detallado de métricas en cada epoch.
        - Manejo de chunks largos y batches grandes para GPU.

    3. Predicción confiable:
        - Resultado binario (1 = listo para entrar, 0 = requiere reforma).

    Attributes:
        - model_name (str): nombre del modelo LLM preentrenado.
        - max_length (int): longitud máxima de la entrada del modelo.
        - embedding_model (str): nombre del modelo de embeddings para pseudo-labeling.
        - device (str): dispositivo en el que se encuentra el modelo (GPU o CPU).
        - tokenizer (AutoTokenizer): tokenizador para la entrada del modelo.
        - model (AutoModelForSequenceClassification): modelo LLM preentrenado.
        - embedding_model (SentenceTransformer): modelo de embeddings para pseudo-labeling.
        - df (pd.DataFrame): DataFrame con las propiedades y sus características.
    """
    def __init__(self,
                 model_name="google/flan-t5-large",
                 max_length=255,
                 embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        """
        Inicializa el objeto con los parámetros de entrenamiento y carga de los modelos.

        Args:
        - model_name (str): nombre del modelo LLM preentrenado.
        - max_length (int): longitud máxima de la entrada del modelo.
        - embedding_model (str): nombre del modelo de embeddings para pseudo-labeling.
        """
        self.model_name = model_name
        self.max_length = max_length

        # LLM preentrenado
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.cuda.empty_cache() #Liberar cache de cuda
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=2
        ).to(self.device)

        # Embeddings para pseudo-labeling semántico
        self.embedding_model = SentenceTransformer(embedding_model)

        # SpaCy español para lematización
        self.nlp = spacy.load("es_core_news_sm")
        self.df = None

    # ----------------------- Carga de datos -----------------------
    def load_raw_data(self):
        """
        Carga los datos de entrenamiento de Kaggle y los limpia de duplicados y descripciones vacías.

        Devuelve un DataFrame con las columnas "url" y "description".
        """
        files = {"kaggle1": "data/raw/kaggle_idealista_madrid.csv",
                 "kaggle2": "data/raw/kaggle_madrid_idealista.csv"}
        
        for key, file in files.items():
            # Carga el archivo de datos
            df = pd.read_csv(file)
            if key == "kaggle1":
                # Selecciona las columnas de interés
                df = df[["url", "description"]]
            elif key == "kaggle2":
                # Selecciona las columnas de interés y renombra las columnas
                df = df[["Link","Summary"]].rename(columns={"Link": "url", "Summary": "description"})
        
            # Concatena los DataFrames
            if self.df is None:
                self.df = df
            else:
                self.df = pd.concat([self.df, df], ignore_index=True)
        
        # Limpieza de duplicados y descripciones vacías
        self.df.drop_duplicates(subset=["url"], inplace=True)
        self.df.dropna(subset=["description"], inplace=True)
        self.df.reset_index(drop=True, inplace=True)

        logger.info(f"Datos cargados: {len(self.df)} registros")
        return self.df

    # ----------------------- Pseudo-labeling profesional -----------------------
    def lemmatize_text(self, text):
        """
        Lematiza el texto utilizando el modelo de lenguaje SpaCy.

        Args:
        - text (str): texto a lematizar.

        Returns:
        - list[str]: lista de lemas del texto.
        """
        doc = self.nlp(str(text).lower())
        return [token.lemma_ for token in doc if token.is_alpha]

    def pseudo_label(self, row):
        """
        Devuelve 1 si la descripción indica 'listo para entrar',
        0 si indica 'requiere reforma', usando embeddings promedio de ejemplos.

        Args:
        - row (pd.DataFrame): fila del DataFrame con la descripción.

        Returns:
        - int: etiqueta binaria (1 si es listo para entrar, 0 si requiere reforma).
        """
        desc = str(row["description"])

        # Listados representativos de ejemplos por clase
        ready_texts = [
            "listo para entrar", "recién reformado", "nuevo", "moderno",
            "sin necesidad de obras", "totalmente reformado", "perfecto estado",
            "reformado recientemente", "acabados modernos", "renovado"
        ]
        reform_texts = [
            "requiere reforma", "a reformar", "antiguo", "viejo", "obsoleto",
            "necesita renovación", "en mal estado", "reformar", "obras pendientes", "necesita reparaciones",
            "para inversores", "con inquilino", "piso ocupado", "okupa"
        ]

        # Embeddings promediados para cada clase
        ready_emb = self.embedding_model.encode(ready_texts, convert_to_tensor=True).mean(dim=0)
        reform_emb = self.embedding_model.encode(reform_texts, convert_to_tensor=True).mean(dim=0)
        desc_emb = self.embedding_model.encode(desc, convert_to_tensor=True)

        # Similitud coseno
        sim_ready = util.cos_sim(desc_emb, ready_emb).item()
        sim_reform = util.cos_sim(desc_emb, reform_emb).item()

        # Retorna etiqueta binaria
        return 1 if sim_ready > sim_reform else 0

    # ----------------------- Generación JSON para fine-tuning -----------------------
    def generate_finetune_json(self, output_file="data/finetune_dataset.json"):
        """
        Genera un archivo JSON para fine-tuning con los datos tokenizados.

        Args:
        output_file (str): Ruta del archivo JSON a generar.

        Returns:
        datasets (DatasetDict): Dataset con los datos tokenizados.
        """
        if self.df is None:
            raise ValueError("Primero carga los datos con load_raw_data()")

        # Convertir DataFrame a Hugging Face Dataset
        dataset = Dataset.from_pandas(self.df)
        dataset_train, dataset_test = dataset.train_test_split(test_size=0.15, seed=42).values()
        datasets = DatasetDict({"train": dataset_train, "test": dataset_test})

        # Función de tokenización con overflow
        def tokenize_function(batch):
            """
            Tokeniza cada descripción y devuelve los prompts, completions y estado binarios.

            Args:
            batch (dict): Diccionario con las descripciones y sus índices.

            Returns:
            dict: Diccionario con los prompts, completions y estado binarios.
            """
            prompts, completions, estado_binarios = [], [], []
            input_ids_list, attention_masks = [], []

            for desc, row_idx in zip(batch["description"], range(len(batch["description"]))):
                estado_binario = self.pseudo_label(self.df.iloc[row_idx])
                estado = "listo para entrar" if estado_binario == 1 else "requiere reforma"

                tokenized = self.tokenizer(
                    desc,
                    truncation=True,
                    max_length=self.max_length,
                    return_overflowing_tokens=True,
                    return_length=True,
                    add_special_tokens=True
                )

                for i, input_ids in enumerate(tokenized["input_ids"]):
                    attention_mask = tokenized["attention_mask"][i]
                    chunk_text = self.tokenizer.decode(input_ids, skip_special_tokens=True)

                    prompts.append(f"descripción: {chunk_text}")
                    completions.append(f" estado: {estado}")
                    estado_binarios.append(estado_binario)
                    input_ids_list.append(input_ids)
                    attention_masks.append(attention_mask)

            return {
                "prompt": prompts,
                "completion": completions,
                "estado_binario": estado_binarios,
                "input_ids": input_ids_list,
                "attention_mask": attention_masks
            }

        # Aplicar map con batching y multi-procesamiento opcional
        for split in ["train", "test"]:
            datasets[split] = datasets[split].map(
                tokenize_function,
                batched=True,
                batch_size=32,
                remove_columns=datasets[split].column_names
            )

        # Guardar todo en un único JSON
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        datasets.save_to_disk("data/tokenized_dataset")

        logger.info(f"Archivo JSON generado: {output_file}")
        return datasets

    # ----------------------- Fine-tuning con early stopping + GPU -----------------------
    def fine_tune(self, tokenized_dataset, output_dir="models", epochs=10, batch_size=1, patience=2):
        """
        Entrena un modelo de clasificación secuencial con early stopping y utiliza la GPU si está disponible.

        Args:
        - tokenized_dataset (DatasetDict): Dataset con los datos tokenizados.
        - output_dir (str): Directorio donde se guardará el modelo entrenado.
        - epochs (int): Número de epochs para entrenar.
        - batch_size (int): Tamaño del batch para entrenar.
        - patience (int): Número de epochs sin mejorar antes de activar early stopping.

        Returns:
        None
        """
        os.makedirs(output_dir, exist_ok=True)

        logger.info(f"Entrenando en dispositivo: {self.device}")

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=1,  # entrenamos manualmente epoch x epoch
            per_device_train_batch_size=batch_size,
            save_strategy="epoch",
            logging_strategy="epoch",
            learning_rate=5e-5,
            weight_decay=0.01,
            report_to="none",
            load_best_model_at_end=False
        )
        self.model.gradient_checkpointing_enable()


        # Data collator para padding dinámico
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["test"],
            tokenizer=self.tokenizer,
            data_collator=data_collator
        )

        best_eval_loss = float("inf")
        epochs_no_improve = 0
        best_model_dir = os.path.join(output_dir, "best_model")
        os.makedirs(best_model_dir, exist_ok=True)

        for epoch in tqdm(range(epochs), desc="Entrenando epochs"):
            logger.info(f"==== Inicio epoch {epoch+1} ====")
            trainer.train()
            eval_result = trainer.evaluate()
            eval_loss = eval_result["eval_loss"]
            logger.info(f"Epoch {epoch+1} eval_loss: {eval_loss:.4f}")

            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                epochs_no_improve = 0
                self.model.save_pretrained(best_model_dir)
                self.tokenizer.save_pretrained(best_model_dir)
                logger.info(f"Mejor modelo actualizado en epoch {epoch+1}")
            else:
                epochs_no_improve += 1
                logger.info(f"No mejora durante {epochs_no_improve} epoch(s)")
                self.model.save_pretrained(output_dir)
                if epochs_no_improve >= patience:
                    logger.info("Early stopping activado")
                    break

# ----------------------- Uso -----------------------
if __name__ == "__main__":
    llm = TrainClassificationLLM()

    llm.load_raw_data()
    llm.generate_finetune_json(output_file="data/finetune_dataset.json")

    # Si ya has extraido el json previamente
    tokenized_dataset = load_from_disk("data/tokenized_dataset")
    tokenized_dataset = tokenized_dataset.rename_column("estado_binario", "labels")
    llm.fine_tune(tokenized_dataset, output_dir="models", epochs=5, batch_size=1, patience=2)