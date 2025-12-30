from sentence_transformers import SentenceTransformer
import torch


class SentenceTokenizerProvider:
    """
    Lightweight tokenizer provider without embedding overhead.
    """

    _tokenizer = None

    @classmethod
    def get_tokenizer(cls, model_name: str = "all-MiniLM-L6-v2"):
        if cls._tokenizer is None:
            # load model ONCE, CPU is enough for tokenizer
            model = SentenceTransformer(
                model_name,
                device="cpu",
            )
            cls._tokenizer = model.tokenizer

        return cls._tokenizer
