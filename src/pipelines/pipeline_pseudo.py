import json
import pandas as pd
import spacy
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util

class PseudoLabeling:
    """
    Clase para calcular la columna 'Reformado' de propiedades.
    Primero filtra por lexemas y luego confirma con embeddings solo los casos necesarios.
    """

    def __init__(self, df: pd.DataFrame, 
                 model: str = "mrm8488/distiluse-base-multilingual-cased-v2-finetuned-stsb_multi_mt-es",
                 ready_threshold: float = -0.1):
        self.df = df.copy()
        self.ready_threshold = ready_threshold

        # Cargar SpaCy
        try:
            self.nlp = spacy.load("es_core_news_sm")
        except OSError:
            raise ValueError("⚠️ Debes instalar el modelo SpaCy: `python -m spacy download es_core_news_sm`")

        # Cargar modelo de embeddings
        self.embedding_model = SentenceTransformer(model)

        # Calcular columna 'Reformado'
        tqdm.pandas(desc="Calculando columna 'Reformado'")
        # Paso 1: lexema
        self.df["Reformado"] = self.df.progress_apply(self._label_lemma, axis=1)
        # Paso 2: embeddings solo en registros con lexema = 1
        mask = self.df["Reformado"] == 1
        if mask.any():
            self.df.loc[mask, "Reformado"] = self.df.loc[mask].progress_apply(self._label_embedding, axis=1)

    # ----------------------- Lemmatización -----------------------
    def _lemmatize_text(self, text: str):
        """Devuelve lista de lemas de un texto."""
        doc = self.nlp(str(text).lower())
        return [token.lemma_ for token in doc if token.is_alpha]

    # ----------------------- Filtrado rápido por lexema -----------------------
    def _label_lemma(self, row):
        desc = str(row.get("Descripción", "")).strip().lower()
        if not desc:
            return 0
        lemmas = self._lemmatize_text(desc)
        return 1 if any("reform" in lemma for lemma in lemmas) else 0

    # ----------------------- Confirmación por embeddings -----------------------
    def _label_embedding(self, row):
        desc = str(row.get("Descripción", "")).strip().lower()
        if not desc:
            return 0
        lemmas = self._lemmatize_text(desc)
        desc_text = " ".join(lemmas)

        ready_texts = [
            "reformado", "reforma integral", "recién reformado", "renovado", "actualizado",
            "listo para entrar", "perfecto estado", "totalmente reformado"
        ]

        desc_emb = self.embedding_model.encode(desc_text, convert_to_tensor=True)
        ready_embs = self.embedding_model.encode(ready_texts, convert_to_tensor=True)
        
        sims = util.cos_sim(desc_emb, ready_embs)  # tensor de similitudes
        sim_ready = sims.max().item()

        return 1 if sim_ready >= self.ready_threshold else 0



if __name__ == "__main__":

    with open("output/idealista_registros.json", 'r', encoding='utf-8') as f:
        data = json.load(f)
    df = pd.DataFrame(data)

    labeling = PseudoLabeling(df)

    df = labeling.df.copy()