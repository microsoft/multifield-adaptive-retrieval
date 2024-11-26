from typing import Tuple, Optional

import sentence_transformers
from transformers import AutoTokenizer, AutoModel, PreTrainedTokenizer, PreTrainedModel, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Normalize, Pooling

def prepare_model(
        model_id: str,
        with_decoder: bool = False,
        normalize: bool = False,
) -> Tuple[PreTrainedTokenizer, SentenceTransformer, Optional[PreTrainedModel]]:
    if model_id.startswith("sentence-transformers/gtr-t5"):
        model = SentenceTransformer(model_id)
        tokenizer = model.tokenizer

        # remove the last normalizer
        if not normalize and isinstance(model._last_module(), Normalize):
            last_id = max(model._modules.keys())
            model._modules.pop(last_id)

        if with_decoder:
            t5_size = model_id.split("-")[-1]
            full_t5 = T5ForConditionalGeneration.from_pretrained(f"google-t5/t5-{t5_size}")
            full_t5.encoder = model._first_module().auto_model.encoder

        return tokenizer, model, full_t5 if with_decoder else None

    elif model_id.startswith("facebook/contriever"):
        model = sentence_transformers.models.Transformer(model_id)
        tokenizer = model.tokenizer
        encoder = model.auto_model
        modules = [
            model,
            Pooling(encoder.config.hidden_size, pooling_mode_mean_tokens=True),
        ]
        if normalize:
            modules.append(Normalize())
        return tokenizer, SentenceTransformer(modules=modules), None

    raise ValueError(f"Unsupported model_id: {model_id}")
