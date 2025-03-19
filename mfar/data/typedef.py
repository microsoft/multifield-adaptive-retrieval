from enum import Enum
from random import random
from typing import Any, Iterator, Optional, Dict, List, Tuple
from dataclasses import dataclass
import gzip
from mashumaro.mixins.json import DataClassJSONMixin

from mfar.data.util import SpecialToken
from mfar.data import trec
from mfar.data.format import format_stark
import json

@dataclass
class Query(DataClassJSONMixin):
    _id: str
    text: str
    metadata: Any = None

    @classmethod
    def from_gzipped(cls, path: str) -> Iterator["Query"]:
        with gzip.open(path, "rt") as f:
            for line in f:
                yield cls.from_json(line.strip())

    def adorn(self) -> "Query":
        return Query(self._id, f"{SpecialToken.query_start.value} {self.text}", self.metadata)

# Should do things properly with tokenization but for now let's not cut anything off.
AVG_WORD_PER_TOKEN = 0.75


@dataclass
class Document(DataClassJSONMixin):
    _id: str
    text: str
    title: Optional[str] = None
    metadata: Any = None

    @classmethod
    def from_gzipped(cls, path: str) -> Iterator["Document"]:
        with gzip.open(path, "rt") as f:
            for line in f:
                yield cls.from_json(line.strip())

    def adorn(self) -> "Document":
        return Document(self._id, f"{SpecialToken.doc_start.value} {self.text}", self.title, self.metadata)

    def random_chunk(self, max_length: int, dataset_name: str = None) -> "Document":
        try:
            words = self.text.split(' ')
        except:
            # It's a stark dataset prob
            if dataset_name is not None:
                _, words = format_stark((0, self.text), dataset_name)
                words = words.split(' ')
            else:
                raise NotImplementedError(f"Dataset {dataset_name} is not supported!")

        max_len_words = max(int(max_length * AVG_WORD_PER_TOKEN), 1)
        if len(words) <= max_len_words:
            if dataset_name is not None:
                return Document(self._id, ' '.join(words), self.title, self.metadata)
            else:
                return self
        start = int(random() * (len(words) - max_len_words))

        return Document(self._id, ' '.join(words[start:start + max_len_words]), self.title, self.metadata)

class FieldType(Enum):
    SPARSE = 1
    DENSE = 2

class Field:
    def __init__(
            self,
            key: str,
            name: str,
            field_type: FieldType,
            max_seq_length: int = 512,
            dataset=None,
        ):
        self.key = key
        self.name = name
        self.field_type = field_type
        self.max_seq_length = max_seq_length
        self.dataset = dataset

    def serialize(self):
        return {
            "key": self.key,
            "name": self.name,
            "field_type": self.field_type.name,
            "max_seq_length": self.max_seq_length,
            "dataset": self.dataset
        }

    @classmethod
    def deserialize(self, data):
        return Field(
            data["key"],
            data["name"],
            FieldType[data["field_type"]],
            data["max_seq_length"],
            data["dataset"]
        )

    def __str__(self):
        return json.dumps(self.__dict__())

    def __dict__(self):
        return {
            # "key": self.key, # Hiding this one because it should always be same as the key
            "name": self.name,
            "field_type": self.field_type.name,
            "max_seq_length": self.max_seq_length
        }

    def __copy__(self):
        return Field(self.key, self.name, self.field_type, self.max_seq_length, self.dataset)

    def __deepcopy__(self, memo):
        return Field(self.key, self.name, self.field_type, self.max_seq_length, self.dataset)


@dataclass
class Corpus:
    docs: List[Document]
    dataset_name: str

    def __post_init__(self):
        self.key_to_id = {doc._id: i for i, doc in enumerate(self.docs)}

    def keys(self) -> Iterator[str]:
        return (doc._id for doc in self.docs)

    def __len__(self):
        return len(self.docs)

    def get_text_by_id(self, doc_id: int) -> str:
        return self.docs[doc_id].text

    def get_text_by_key(self, key: str) -> str:
        return self.docs[self.key_to_id[key]].text

    def get_doc_by_id(self, doc_id: int) -> Document:
        return self.docs[doc_id]

    def get_doc_by_key(self, key: str) -> Document:
        try:
            return self.docs[self.key_to_id[key]]
        except KeyError:
            raise KeyError(f"Key {key} not found in corpus.")

    def pairs(self):
        return ((doc._id, doc.text) for doc in self.docs)

    @classmethod
    def from_trec(cls, path: str) -> "Corpus":
        docs = [
            Document(key, text)
            for key, text in trec.read_corpus(path)
        ]
        return cls(docs)


    @classmethod
    def from_docs_dict(cls, docs_dict: Dict[int, str], dataset_name: str = None) -> "Corpus":
        docs = [
            Document(key, text)
            for key, text in docs_dict.items()
        ]
        return cls(docs, dataset_name)
