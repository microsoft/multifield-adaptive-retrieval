from concurrent.futures import ThreadPoolExecutor
from typing import Tuple, Optional
from pathlib import Path
import numpy as np

from mfar.data import trec
from mfar.data.format import format_documents
from mfar.data.index import BM25sSparseIndex, DenseFlatIndex
from mfar.data.typedef import Corpus, FieldType
from mfar.data.util import MemoryMapDict
import sentence_transformers
from transformers import PreTrainedTokenizer, PreTrainedModel, T5ForConditionalGeneration, AutoModel
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Normalize, Pooling

def prepare_model(
        model_id: str,
        with_decoder: bool = False,
        normalize: bool = False,
        freeze_encoder: bool = False,
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
        if freeze_encoder:
            for param in encoder.parameters():
                param.requires_grad = False
            encoder.training = False
        modules = [
            model,
            Pooling(encoder.config.hidden_size, pooling_mode_mean_tokens=True),
        ]
        if normalize:
            modules.append(Normalize())
        return tokenizer, SentenceTransformer(modules=modules), None

    # Else maybe it is a directory, try to load model
    else:
        try:
            model = sentence_transformers.models.Transformer(model_id)
        except:
            raise ValueError(f"Unsupported model_id or unable to find: {model_id}")
        tokenizer = model.tokenizer
        encoder = model.auto_model
        if freeze_encoder:
            for param in encoder.parameters():
                param.requires_grad = False
        modules = [
            model,
            Pooling(encoder.config.hidden_size, pooling_mode_mean_tokens=True),
        ]
        if normalize:
            modules.append(Normalize())
        return tokenizer, SentenceTransformer(modules=modules), None

def read_and_create_indices(corpus_path, dataset_name, field_info, temp_dir, encoder):
    corpus = list(trec.read_corpus(corpus_path))
    vectors_dict = {}
    indices_dict = {}

    if any(field.field_type == FieldType.DENSE for field in field_info.values()):
        # These values are not defined in the sparse case, but they are also unused so an exception should be raised
        numeric_ids_to_keys = [x[0] for x in corpus]
        keys_to_numeric_ids = {k: i for i, k in enumerate(numeric_ids_to_keys)}

    for field_key, field in field_info.items():
        if field.field_type == FieldType.DENSE:
            v_files = f"{temp_dir}/{field.name}.npy"
            Path(v_files).parents[0].mkdir(parents=True, exist_ok=True)
            # Create a file at the locations (somewhat hacky)
            with open(v_files, 'w') as fp:
                pass
            vectors = MemoryMapDict(
                v_files,
                keys=[x[0] for x in corpus],
                shape=(len(corpus), encoder.get_sentence_embedding_dimension()),
            )  # shared across GPUs
            vectors_dict[field_key] = vectors
            indices_dict[field_key] = DenseFlatIndex(
                encoder,
                vectors.file,
                numeric_ids_to_keys=numeric_ids_to_keys,
                keys_to_numeric_ids=keys_to_numeric_ids,
            )
        elif field.field_type == FieldType.SPARSE:
            formatted_docs = format_documents(corpus, field.name, field.dataset)
            modified_field_name_docs = Corpus.from_docs_dict({item[0]: item[1] for item in formatted_docs})
            indices_dict[field_key] = BM25sSparseIndex.create(modified_field_name_docs, dataset_name=dataset_name)
            indices_dict[field_key].name = field.name

    return corpus, vectors_dict, indices_dict



def _create_sparse_index_from_npy(keys, vals):
    """
    keys: np array: np.int of size [num_keys, 2]
    vals: np array: np.float of size [num_keys]

    Helper function to parallelize things a lot becuase we can have O(100M) keys that we need to convert into dict
    """
    CHUNK_SIZE = 2**17 # this number can probably change a little but not a lot
    def _insert_tuple(start_idx):
        new_dict = {}
        for offset in range(CHUNK_SIZE):
            if start_idx + offset >= len(keys):
                return new_dict
            key = keys[start_idx + offset]
            val = vals[start_idx + offset]
            if key[0] not in new_dict:
                new_dict[key[0]] = {}
            new_dict[key[0]][key[1]] = val
        return new_dict

    return_dict = {}
    if len(keys) > 0:
        unique_keys = np.unique(keys[:, 0])
    else:
        return {}
    num_chunks = len(keys)//CHUNK_SIZE + 1
    start_idxs = list(range(num_chunks))

    for key in unique_keys:
        return_dict[key] = {}

    with ThreadPoolExecutor(max_workers=min(num_chunks, 64)) as mini_executor:
        new_dicts = list(mini_executor.map(_insert_tuple, start_idxs))
    for partial_result in new_dicts:
        for k, v_dict in partial_result.items():
            return_dict[k].update(v_dict)
    return return_dict


def read_sparse_scores(scores_path, field_info):
    """
    scores_path: path to scores files that look like {scores_path}/{field_name}_keys_bm25.npy and {scores_path}/{field_name}_vals_bm25.npy
    field_info: dict of field_name to field object

    Returns: dict of field_name to sparse scores dict, where the sparse scores dict is {qid: {doc_id: score}}
    """
    def read_sparse_scores_per_field(field_key):
        return_dict = {}
        load_paths = [f"{scores_path}/{field_key}_keys_bm25.npy", f"{scores_path}/{field_key}_vals_bm25.npy"]
        with ThreadPoolExecutor(max_workers=2) as mini_executor:
            keys, vals = list(mini_executor.map(np.load, load_paths))
        assert(len(keys) == len(vals))
        return_dict = _create_sparse_index_from_npy(keys, vals)
        return return_dict
    sparse_fields = [field for field in field_info.keys() if field_info[field].field_type == FieldType.SPARSE]
    # len(sparse_fields) should be greater than 0
    if len(sparse_fields) == 0:
        print("No sparse fields found")
    with ThreadPoolExecutor(max_workers=len(sparse_fields)) as executor:
         results = list(executor.map(read_sparse_scores_per_field, sparse_fields))
    return_dict = {field: result for field, result in zip(sparse_fields, results)}
    return return_dict