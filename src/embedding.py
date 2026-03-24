from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List
import torch


class EmbeddingManager:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", batch_size: int = 128):
        """
        Production-ready embedding manager
        """

        self.model_name = model_name
        self.batch_size = batch_size

        #  Use GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = SentenceTransformer(
            model_name,
            device=self.device
        )

    def _clean_text(self, text: str) -> str:
        """
        Light normalization (preserve semantics)
        """
        if not text:
            return ""

        # Remove excessive whitespace only
        text = " ".join(text.strip().split())
        return text

    def _safe_truncate(self, text: str, max_words: int = 400) -> str:
        """
        Prevent silent truncation by transformer models
        """
        words = text.split()
        return " ".join(words[:max_words])

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings with:
        - cleaning
        - truncation
        - batching
        """

        processed_texts = []

        for t in texts:
            if not t or not t.strip():
                continue  # skip empty safely

            cleaned = self._clean_text(t)
            truncated = self._safe_truncate(cleaned)

            processed_texts.append(truncated)

        if not processed_texts:
            return np.array([])

        embeddings = self.model.encode(
            processed_texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True ,
            convert_to_numpy=True
        )

        return np.array(embeddings)