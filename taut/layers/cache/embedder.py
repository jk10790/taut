"""Embedder interface for semantic caching."""
from abc import ABC, abstractmethod

class Embedder(ABC):
    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Embed a single text string into a vector."""
        ...

    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""
        ...

class ONNXEmbedder(Embedder):
    def __init__(self, model_name: str = "Xenova/all-MiniLM-L6-v2"):
        self._session = None
        self._tokenizer = None
        self._model_name = model_name

    def _get_model(self):
        if self._session is None:
            try:
                import onnxruntime as ort
                from tokenizers import Tokenizer
                import huggingface_hub
            except ImportError:
                raise ImportError(
                    "ONNX backend dependencies are missing. "
                    "Install with `pip install taut[cache]`."
                ) from None
            
            # Using huggingface_hub to download ONNX model and tokenizers
            model_path = huggingface_hub.hf_hub_download(repo_id=self._model_name, filename="onnx/model.onnx")
            tokenizer_path = huggingface_hub.hf_hub_download(repo_id=self._model_name, filename="tokenizer.json")
            
            self._session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
            self._tokenizer = Tokenizer.from_file(tokenizer_path)
            
        return self._session, self._tokenizer

    async def embed(self, text: str) -> list[float]:
        import asyncio
        import concurrent.futures
        
        loop = asyncio.get_running_loop()
        
        def _compute():
            import numpy as np
            session, tokenizer = self._get_model()
            encoded = tokenizer.encode(text)
            
            # ONNX models from Xenova usually require these 3 inputs
            inputs = {
                "input_ids": np.array([encoded.ids], dtype=np.int64),
                "attention_mask": np.array([encoded.attention_mask], dtype=np.int64),
            }
            if "token_type_ids" in [i.name for i in session.get_inputs()]:
                inputs["token_type_ids"] = np.array([encoded.type_ids], dtype=np.int64)
                
            outputs = session.run(None, inputs)
            embeddings = outputs[0]
            
            # Mean pooling
            mask = inputs["attention_mask"][..., np.newaxis]
            sum_embeddings = np.sum(embeddings * mask, axis=1)
            sum_mask = np.clip(np.sum(mask, axis=1), a_min=1e-9, a_max=None)
            pooled = sum_embeddings / sum_mask
            
            # Normalize
            norm = np.linalg.norm(pooled, axis=1, keepdims=True)
            return (pooled / norm)[0].tolist()

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return await loop.run_in_executor(pool, _compute)

    def dimension(self) -> int:
        return 384
