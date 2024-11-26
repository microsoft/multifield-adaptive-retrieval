from typing import Iterable, Dict, Generic, Sequence, Tuple, TypeVar, Literal, List, Union, Optional, Any
from abc import ABC, abstractmethod

import numpy as np
import json
import torch
import bm25s

from Stemmer import Stemmer

from tqdm import tqdm
from more_itertools import chunked
from sentence_transformers import SentenceTransformer

from more_itertools import chunked
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from mfar.data.typedef import Corpus

Key = TypeVar("Key")
Query = TypeVar("Query")
Vector = TypeVar("Vector")


class Index(ABC, Generic[Key, Query]):
    """
    Encapsulates any index that can be searched over.
    It can either be a sparse index (e.g. powered by Whoosh), or a dense index (e.g. powered by FAISS).
    """

    @abstractmethod
    def retrieve(self, query: Query, top_k: int) -> Sequence[Tuple[Key, float]]:
        raise NotImplementedError

    def retrieve_batch(
        self, queries: Sequence[Query], top_k: int
    ) -> Sequence[Sequence[Tuple[Key, float]]]:
        """
        May be overridden by subclasses to provide a more efficient implementation.
        """
        return [self.retrieve(q, top_k) for q in queries]

class BM25sSparseIndex(Index[str, str]):
    """
    A sparse index powered by BM25s (without Lucene/JVMs).
    """

    def __init__(self, keys: List[str], index: bm25s.BM25, stemmer: Optional[Stemmer], index_limit: int = 5000):
        self.keys = keys
        self.key_to_id = {key: i for i, key in enumerate(keys)}
        self.index = index
        self.stemmer = stemmer
        self.index_memo = {}        # str -> Index thing
        self.tokenization_memo = {} # str -> List[str]
        self.index_limit = index_limit

    def tokenize_single(self, query, stopwords, stemmer, return_ids):
        if query in self.tokenization_memo:
            return self.tokenization_memo[(query, return_ids)]
        tokens = bm25s.tokenize(query, stopwords=stopwords, stemmer=stemmer, return_ids=return_ids, show_progress=False)
        if return_ids:
            self.tokenization_memo[(query, return_ids)] = tokens
        else:
            self.tokenization_memo[(query, return_ids)] = tokens[0]
        return self.tokenization_memo[(query, return_ids)]

    def tokenize(self, queries, stopwords, stemmer, return_ids=True):
        if isinstance(queries, str):
            return self.tokenize_single(queries, stopwords, stemmer, return_ids)
        return [self.tokenize_single(query, stopwords, stemmer, return_ids) for query in queries]

    def get_scores(self, query):
        if query in self.index_memo:
            return self.index_memo[query]
        query_tokens = self.tokenize(query, stopwords="en", stemmer=self.stemmer, return_ids=False)
        score = self.index.get_scores(query_tokens)
        if len(self.index_memo) < 5000:
            self.index_memo[query] = score
        return score

    def retrieve(self, query: str, top_k: int) -> Sequence[Tuple[str, float]]:
        query_tokens = self.tokenize(query, stopwords="en", stemmer=self.stemmer, return_ids=False)
        results, scores = self.index.retrieve([query_tokens], k=top_k, show_progress=False)  # [Batch, Cand]
        return [(self.keys[results[0, i]], scores[0, i]) for i in range(results.shape[1])]

    def retrieve_batch(
        self, queries: Sequence[str], top_k: int
    ) -> Sequence[Sequence[Tuple[str, float]]]:
        query_tokens = self.tokenize(queries, stopwords="en", stemmer=self.stemmer, return_ids=False)
        results, scores = self.index.retrieve(query_tokens, k=top_k, show_progress=False)
        return [
            [(self.keys[results[i, j]], scores[i, j]) for j in range(results.shape[1])]
            for i in range(results.shape[0])
        ]

    def score(self, query: str, keys: Sequence[str]) -> np.ndarray:  # [Cand]
        doc_ids = np.array([self.key_to_id[key] for key in keys])
        all_doc_scores = self.get_scores(query)  # [D]
        specified_doc_scores = all_doc_scores[doc_ids]  # [Cand]
        return specified_doc_scores

    def score_batch(self, queries: Sequence[str], keys: Sequence[str]) -> np.ndarray:  # [Query, Cand]
        doc_ids = np.array([self.key_to_id[key] if key in self.key_to_id else -1 for key in keys])
        neg_indices = [idx for idx, item in enumerate(doc_ids) if item == -1]
        all_doc_scores = [self.get_scores(query) for query in queries]
        all_doc_scores = np.stack(all_doc_scores, axis=0)  # [Query, D]
        specified_doc_scores = all_doc_scores[:, doc_ids]
        specified_doc_scores[:, neg_indices] = 0
        return torch.tensor(specified_doc_scores)

    def get_scores_with_dummy_check(self, func, default=None, *args, **kwargs):
        try:
            breakpoint()
            return func(*args, **kwargs)
        except Exception:
            return 0

    @classmethod
    def create(cls, corpus: Corpus, stemmer: Optional[Stemmer] = None, dataset_name: Optional[str] = ""):
        keys = list(corpus.keys())
        texts = [d.text for d in corpus.docs]
        index = bm25s.BM25(method="lucene", k1=1.2, b=0.75)
        doc_tokens = bm25s.tokenize(texts, stopwords="en", stemmer=stemmer, show_progress=False)
        index.index(doc_tokens, show_progress=False)
        if dataset_name == "amazon":
            index_limit = 5000
        else:
            index_limit = 12000
        return cls(keys, index, stemmer, index_limit)

    def save(self, path: str):
        self.index.save(f"{path}/index")
        with open(f"{path}/keys.json", "w") as f:
            json.dump(self.keys, f)

    @classmethod
    def load(cls, path: str, stemmer: Optional[Stemmer] = None):
        with open(f"{path}/keys.json", "r") as f:
            keys = json.load(f)
        index = bm25s.BM25.load(f"{path}/index", mmap=True)
        return cls(keys, index, stemmer)


class DenseFlatIndex(Index[str, str]):

    def __init__(
            self,
            model: SentenceTransformer,
            vectors: np.memmap,
            numeric_ids_to_keys: Sequence[str],
            keys_to_numeric_ids: Dict[str, int],
            device: torch.device = torch.device("cpu"),
            vector_batch_size: int = 1048576,
    ):
        self.model = model
        self.vectors = vectors
        self.numeric_ids_to_key = numeric_ids_to_keys
        self.key_to_numeric_ids = keys_to_numeric_ids
        self.device = device
        self.vector_batch_size = vector_batch_size

    def retrieve(self, query: Query, top_k: int) -> Sequence[Tuple[Key, float]]:
        return self.retrieve_batch([query], top_k)[0]

    def retrieve_batch(
        self, queries: Union[np.ndarray, Sequence[Query]], top_k: int,
    ) -> Sequence[Sequence[Tuple[str, float]]]:
        if isinstance(queries, np.ndarray):
            query_encodings = torch.from_numpy(queries).to(device=self.device)
        else:
            query_encodings = self.model.encode(list(queries), convert_to_tensor=True).to(device=self.device)

        batch_size = query_encodings.size(0)
        vector_batch_size = self.vector_batch_size

        top_scores = torch.zeros((batch_size, top_k), dtype=torch.float32, device=self.device)
        top_indices = torch.zeros((batch_size, top_k), dtype=torch.int64, device=self.device)
        for lb in range(0, self.vectors.shape[0], vector_batch_size):
            ub = min(self.vectors.shape[0], lb + vector_batch_size)
            vector_batch = torch.from_numpy(self.vectors[lb:ub]).to(device=self.device)
            scores = torch.matmul(query_encodings, vector_batch.t())  # R[Q, B]
            combined_scores = torch.cat([top_scores, scores], dim=1)
            combined_indices = torch.cat([
                top_indices,
                torch.arange(lb, ub, device=self.device).unsqueeze(0).expand(batch_size, -1)
            ], dim=1)
            combined_top_scores, combined_top_indices = torch.topk(
                combined_scores,
                top_k,
                dim=1,
                largest=True,
                sorted=True
            )

            top_indices = combined_indices[torch.arange(batch_size, device=self.device).unsqueeze(1), combined_top_indices]
            top_scores = combined_top_scores[:, :top_k]

        indices_list = top_indices.tolist()
        scores_list = top_scores.tolist()
        return [
            list(zip(
                [self.numeric_ids_to_key[j] for j in indices_list[i]],
                scores_list[i]
            ))
            for i in range(len(queries))
        ]

    def score(self, query: Query, keys: Sequence[str]) -> Sequence[Tuple[Key, float]]:
        return self.score_batch([query], [keys])[0]

    def score_batch(self, queries: Sequence[Query], keys: Sequence[Sequence[str]]) -> Sequence[Sequence[Tuple[str, float]]]:
        query_encodings = self.model.encode(queries, convert_to_tensor=True).to(device=self.device)
        indices_to_keys = [self.key_to_numeric_ids[idx] for idx in keys]
        selected_vectors = torch.from_numpy(self.vectors[indices_to_keys])
        scores = torch.matmul(query_encodings, selected_vectors.t())
        return scores

def candidate_encoding_stream(
        encoder: SentenceTransformer,
        corpus: Iterable[Tuple[str, str]],
        batch_size: int = 64,
        multiprocess: bool = True,
        show_progress: bool = True,
) -> Iterable[Tuple[str, np.ndarray]]:
    if show_progress:
        chunked_iterator = chunked(tqdm(corpus), n=batch_size)
    else:
        chunked_iterator = chunked(corpus, n=batch_size)
    if multiprocess:
        pool = encoder.start_multi_process_pool()
        for batch in chunked_iterator:
            ids = [id for id, _ in batch]
            texts = [text for _, text in batch]
            embs = encoder.encode_multi_process(texts, batch_size=batch_size, pool=pool, chunk_size=batch_size // len(pool['processes']))
            yield from zip(ids, embs)
        encoder.stop_multi_process_pool(pool)
    else:
        for batch in chunked_iterator:
            ids = [id for id, _ in batch]
            texts = [text for _, text in batch]
            embs = encoder.encode(texts, batch_size=batch_size, convert_to_numpy=True)
            yield from zip(ids, embs)
