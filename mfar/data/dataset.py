from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Dict, Set, Optional, Union

from mfar.data.index import Index
from torch.utils.data import Dataset
from transformers import BatchEncoding, PreTrainedTokenizer

from mfar.data.format import format_documents
from mfar.data.negative_sampler import NegativeSampler
from mfar.data.typedef import FieldType, Query, Document, Corpus
from mfar.data import trec

class Kind(Enum):
    QUERY = "query"
    HYBRID = "hybrid"

@dataclass
class Instance:
    mode: Kind
    query: Optional[Query] = None
    pos_cand: Optional[Document] = None
    neg_cands: List[Document] = field(default_factory=list)

    @classmethod
    def create(
            cls,
            mode: Kind,
            q: Optional[Query],
            d: Document,
            corpus: Corpus,
            neg_sampler: Optional[NegativeSampler] = None,
            pos_for_each_qid: Optional[Dict[str, Set[str]]] = None,
            max_length: int = 384,
    ) -> "Instance":
        pos_cand = d.random_chunk(max_length, corpus.dataset_name)
        neg_cands = [
            cand.random_chunk(max_length, corpus.dataset_name)
            for cand in neg_sampler.sample(q, pos_for_each_qid)
        ] if neg_sampler is not None else []

        return Instance(
            mode=mode,
            query=q,
            pos_cand=pos_cand,
            neg_cands=neg_cands,
        )

@dataclass
class DecomposedInstance:
    mode: Kind
    query: Optional[Query] = None
    pos_cand: Optional[Document] = None
    neg_cands: List[Document] = field(default_factory=list)

    @classmethod
    def create(
            cls,
            mode: Kind,
            q: Optional[Query],
            d: Document,
            neg_sampler: Optional[NegativeSampler] = None,
            pos_for_each_qid: Optional[Dict[str, Set[str]]] = None,
            field_info: Optional[Dict] = None,
            random_chunk: bool = True,
    ) -> "DecomposedInstance":
        """
        Takes in multiple instances instead of just one and parses the parts of the document.

        Parameters:
        mode: the mode chosen (e.g. supervised, unsupervised)
        query: the query used for the specific instance
        pos_cand_part: the specific positive candidate, separated into parts
        neg_cand_part: the specific negative candidates sampled, separated into parts

        Returns: a decomposed instance

        """
        neg_cands = [
            (cand._id, cand.text) for cand in neg_sampler.sample(q, pos_for_each_qid)
        ] if neg_sampler is not None else []

        pos_cand_parts = {
            field.key: format_documents([(d._id, d.text)], field.name, field.dataset)[0]
            for field in field_info.values()
        }

        neg_cands = [cand for cand in neg_cands]
        neg_cands_parts = {
            field.key: [format_documents([cands], field.name, field.dataset)[0] for cands in neg_cands]
            for field in field_info.values()
        }

        # For each field name, convert to document, and then
        for field_key in pos_cand_parts:
            items = pos_cand_parts[field_key]
            if random_chunk:
                doc = Document(items[0], items[1]).random_chunk(field_info[field_key].max_seq_length)
            else:
                doc = Document(items[0], items[1])
            pos_cand_parts[field_key] = (doc._id, doc.text)

        for field_key in neg_cands_parts:
            new_items = []
            for item in neg_cands_parts[field_key]:
                if random_chunk:
                    doc = Document(item[0], item[1]).random_chunk(field_info[field_key].max_seq_length)
                else:
                    doc = Document(item[0], item[1])
                new_items.append((doc._id, doc.text))
            neg_cands_parts[field_key] = new_items

        return DecomposedInstance(
            mode=mode,
            query=q,
            pos_cand=pos_cand_parts,
            neg_cands=neg_cands_parts,
        )


@dataclass
class InstanceBatch:
    mode: Kind
    query: Optional[BatchEncoding] = None
    pos_cand: Optional[BatchEncoding] = None
    neg_cands: Optional[BatchEncoding] = None
    instances: List[Union[Instance, Query]] = field(default_factory=list)


@dataclass
class DecomposedInstanceBatch:
    mode: Kind
    query: Optional[BatchEncoding] = None
    pos_cand: Optional[BatchEncoding] = None
    neg_cands: Optional[BatchEncoding] = None
    instances: List[Union[DecomposedInstance, Query]] = field(default_factory=list)

class QueryDataset(Dataset[Query]):
    def __init__(self,
                 tokenizer: PreTrainedTokenizer,
                 queries: Dict[str, str],
                 max_length: int = 512,
                 field_types: Set[FieldType] = None,
                 ):
        self.queries = queries
        self.tokenizer = tokenizer
        self.max_length = max_length # What do we use this for?
        self.ids = list(self.queries.keys())
        self.field_types: Set[FieldType] = field_types
        self.indices_dict: Dict

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx: int) -> Query:
        query_id = self.ids[idx]
        query = self.queries[query_id]
        # Some queries are too short, temporary hack so length of query is nonzero for embeds
        if len(query.strip()) < 5:
            query = "what"
        return Query(query_id, query)

    def collate(self,
                instances: List[Query],
        ) -> InstanceBatch:
        texts = [instance.text for instance in instances]
        query = {}
        query[FieldType.DENSE] = self.tokenizer.batch_encode_plus(
            texts,
            max_length=self.max_length,
            padding='longest',
            return_tensors="pt",
        )

        return InstanceBatch(
            mode=Kind.QUERY,
            query=query,
            instances=instances
        )

class ContrastiveTrainingDataset(Dataset[Instance]):
    def __init__(self,
                 tokenizer: PreTrainedTokenizer,
                 queries: Dict[str, str],
                 documents: Corpus,
                 qrels: List[trec.QRels],
                 negative_sampler: NegativeSampler,
                 num_neg_sample_per_layer: int = 0,
                 max_length: int = 384,
                 field_info: Dict = None,
                 field_types: Set[FieldType] = None,
                 indices_dict: Dict = None,
                 prefix: bool = False,
                 random_chunk: bool = True,
                 ):
        self.tokenizer = tokenizer
        self.queries = queries
        self.documents = documents
        self.qrels = qrels
        self.neg_sampler = negative_sampler
        self.num_neg_sample_per_layer = num_neg_sample_per_layer
        self.max_length = max_length
        self.field_info = field_info
        self.prefix = prefix
        self.random_chunk = random_chunk

        self.pos_for_each_qid: Dict[str, Set[str]] = {}
        for qrel in self.qrels:
            if qrel.query_id not in self.pos_for_each_qid:
                self.pos_for_each_qid[qrel.query_id] = set()
            self.pos_for_each_qid[qrel.query_id].add(qrel.doc_id)

        self.field_types = field_types
        self.indices_dict = indices_dict

    def __len__(self):
        return len(self.qrels)

    def __getitem__(self, idx: int) -> Instance:
        qrel = self.qrels[idx]
        query = Query(qrel.query_id, self.queries[qrel.query_id])
        # This was a temporary hack, sorry it is now permanent
        if len(query.text.strip()) < 5:
            query.text = "what"
        pos_cand = self.documents.get_doc_by_key(qrel.doc_id)

        return DecomposedInstance.create(
            mode=Kind.HYBRID,
            q=query,
            d=pos_cand,
            neg_sampler=self.neg_sampler,
            pos_for_each_qid=self.pos_for_each_qid,
            field_info=self.field_info,
            random_chunk=self.random_chunk,
        )

    def collate(self,
                instances: List[Instance],
                ) -> InstanceBatch:
        def _batch_encode(sentences: List[str], max_length=self.max_length) -> BatchEncoding:
            return self.tokenizer.batch_encode_plus(
                batch_text_or_text_pairs=sentences,
                padding='longest',
                max_length=max_length,
                truncation='longest_first',
                return_tensors="pt",
            )

        query = {}
        pos_cand = {}
        neg_cands = {}
        query[FieldType.DENSE] = _batch_encode([instance.query.text for instance in instances])

        if FieldType.DENSE in self.field_types:
            _pos_cand = {
                field.key: [
                    field.name.replace("___", " ") + ": " + instance.pos_cand[field.key][1]
                    if self.prefix else instance.pos_cand[field.key][1]
                    for instance in instances
                ] for field in self.field_info.values()
            }

            pos_cand[FieldType.DENSE] = {
                field.key: _batch_encode(_pos_cand[field.key], max_length=field.max_seq_length)
                for field in self.field_info.values()
            }

            _neg_cands = {
                field.key: [
                    field.name.replace("___", " ") + ": " + instance.neg_cands[field.key][0][1]
                    if self.prefix else instance.neg_cands[field.key][0][1]
                    for instance in instances
                ] for field in self.field_info.values()
            }

            neg_cands[FieldType.DENSE] = {
                field.key: _batch_encode(_neg_cands[field.key], max_length=field.max_seq_length)
                for field in self.field_info.values()
            }
        else:
            pos_cand[FieldType.DENSE] = []
            neg_cands[FieldType.DENSE] = []

        return DecomposedInstanceBatch(
            mode=Kind.HYBRID,
            query=query,
            pos_cand=pos_cand,
            neg_cands=neg_cands,
            instances=instances
        )

def any_collate(
    dataset: Dataset,
    instances: List[Union[Instance, DecomposedInstance]],
) -> Union[InstanceBatch, DecomposedInstanceBatch]:
    if isinstance(dataset, ContrastiveTrainingDataset):
        return dataset.collate(instances)