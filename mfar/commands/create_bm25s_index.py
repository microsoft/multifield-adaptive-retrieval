from typing import List
from mfar.data.index import BM25sSparseIndex
from mfar.data.format import format_documents
from mfar.data.typedef import Corpus
from mfar.data import trec

from mfar.data.schema import resolve_fields

import fire

def main(
        data_path: str,
        dataset_name: str,
        output_path: str,
        fields_str: str="all_sparse,single_sparse", # By default, index everything so we don't need to do it for different exps
):
    fields = resolve_fields(fields_str, dataset_name)
    corpus = list(trec.read_corpus(f"{data_path}/corpus"))
    for field_name, field in fields.items():
        formatted_documents = format_documents(corpus, field.name, field.dataset)
        formatted_corpus = Corpus.from_docs_dict({item[0]: item[1] for item in formatted_documents},
                                                 dataset_name=dataset_name)
        index = BM25sSparseIndex.create(formatted_corpus, dataset_name=dataset_name)
        index.save(f"{output_path}/{field_name}_sparse_index")

if __name__ == "__main__":
    fire.Fire(main)