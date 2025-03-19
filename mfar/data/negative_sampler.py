from abc import ABC
from typing import AbstractSet, List, Set, Mapping, Tuple

import random

from mfar.data.index import Index
from mfar.data.typedef import Query, Document


class NegativeSampler(ABC):
    @property
    def n_sample(self) -> int:
        raise NotImplementedError

    def sample(self, query: Query, pos_for_each_qid: Mapping[str, AbstractSet[str]]) -> List[Document]:
        raise NotImplementedError

    def sample_batch(self, queries: List[Query], pos_for_each_qid: Mapping[str, AbstractSet[str]]) -> List[List[Document]]:
        raise NotImplementedError


class IndexNegativeSampler(NegativeSampler):
    def __init__(self,
                 index: Index,
                 documents: Mapping[str, str],
                 n_retrieve: int = 50,
                 n_bottom: int = 5,
                 n_sample: int = 1,
                 ):
        self.index = index
        self.documents = documents
        self.n_retrieve = n_retrieve
        self.n_bottom = n_bottom
        self._n_sample = n_sample

    @property
    def n_sample(self) -> int:
        return self._n_sample

    def sample(self, query: Query, pos_for_each_qid: Mapping[str, AbstractSet[str]]) -> List[Document]:
        neg_cand_with_scores: List[Tuple[str, float]] = [
            (doc_id, score)
            for doc_id, score in self.index.retrieve(query.text, top_k=self.n_retrieve)
            if doc_id not in pos_for_each_qid[query._id]  # remove correct samples
        ]
        if len(neg_cand_with_scores) == 0:
            new_n_retrieve = len(pos_for_each_qid[query._id]) + self.n_bottom
            neg_cand_with_scores = [
                (doc_id, score)
                for doc_id, score in self.index.retrieve(query.text, top_k=new_n_retrieve)
                if doc_id not in pos_for_each_qid[query._id]  # remove correct samples
            ]
        neg_cand_with_scores.sort(key=lambda x: x[1], reverse=True)
        neg_cand_ids = [doc_id for doc_id, _ in neg_cand_with_scores[-self.n_bottom:]]
        sampled_neg_cand_ids = [neg_cand_ids[i] for i in random.sample(range(len(neg_cand_ids)), self.n_sample)]
        sampled_neg_cands = [
            Document(i, self.documents.get(i, ""))
            for i in sampled_neg_cand_ids
        ]
        return sampled_neg_cands

    def sample_batch(self, queries: List[Query], pos_for_each_qid: Mapping[str, AbstractSet[str]]) -> List[List[Document]]:
        # TODO: implement batch sampling
        return [self.sample(q, pos_for_each_qid) for q in queries]