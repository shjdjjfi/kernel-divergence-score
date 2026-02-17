import hashlib
import math
from dataclasses import dataclass


class LiteTokenizer:
    def __call__(self, texts, padding=True, truncation=True, return_tensors=None):
        if isinstance(texts, str):
            texts = [texts]
        input_ids = []
        attention_mask = []
        for text in texts:
            toks = [ord(c) % 256 for c in text][:128]
            if not toks:
                toks = [0]
            input_ids.append(toks)
            attention_mask.append([1] * len(toks))
        return {"input_ids": input_ids, "attention_mask": attention_mask}


@dataclass
class LiteModelState:
    trained: bool = False
    train_shift: float = 0.0


class LiteModelWrapper:
    def __init__(self, args, model_name):
        self.name = model_name
        self.tokenizer = LiteTokenizer()
        self.state = LiteModelState()

    def fit(self, dataset, contamination):
        self.state.trained = True
        self.state.train_shift = float(contamination)

    def _vector_for_text(self, text, layer_idx, dim=32):
        vals = []
        suffix = f"|layer={layer_idx}|trained={self.state.trained}|shift={self.state.train_shift:.4f}"
        for i in range(dim):
            digest = hashlib.sha256(f"{text}|{i}{suffix}".encode()).digest()
            n = int.from_bytes(digest[:8], "big")
            vals.append((n / (2**64 - 1)) * 2.0 - 1.0)
        norm = math.sqrt(sum(v * v for v in vals)) or 1.0
        return [v / norm for v in vals]

    def __call__(self, batch, output_hidden_states=True, hidden_states_layers_to_output=None):
        texts = batch.get("raw_texts", [])
        layers = []
        for layer_idx in range(33):
            layer_vectors = [self._vector_for_text(t, layer_idx) for t in texts]
            layers.append(layer_vectors)
        return layers, None, None, None
