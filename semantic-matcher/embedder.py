"""BGE-M3 embedding logic using sentence-transformers."""

from typing import Optional
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-base-en-v1.5"
VECTOR_SIZE = 768

_model: Optional[SentenceTransformer] = None


def get_model() -> SentenceTransformer:
    """Lazy-load the BGE-M3 model (downloaded on first run, ~500MB)."""
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_text(text: str) -> list[float]:
    """Embed a single text string, returning a 1024-dim vector."""
    model = get_model()
    vector = model.encode(text, normalize_embeddings=True)
    return vector.tolist()


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed multiple texts in one batch."""
    model = get_model()
    vectors = model.encode(texts, normalize_embeddings=True)
    return [v.tolist() for v in vectors]
