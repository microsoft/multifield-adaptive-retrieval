from typing import Optional
from tqdm import tqdm

from stark_qa import load_qa
from stark_qa.retrieval import STaRKDataset

from mfar.data import trec
from mfar.data.util import remove_irregularities

from fire import Fire
import os
import json

STARK_DATASETS = ["mag", "prime", "amazon"]

def main(*, dataset_name: str, out: str, max_docs: Optional[int] = None):
    print(f"Downloading queries for {dataset_name} to {out}, with max_docs={max_docs}.")
    if max_docs is not None:
        if int(max_docs) == -1:
            max_docs = None
            print(f"Setting max_docs to None")
        else:
            max_docs = int(max_docs)
            print(f"Setting max_docs to {max_docs} (with boolean value {bool(max_docs)}).")
    qa: STaRKDataset = load_qa(name=dataset_name)
    for partition in ["train", "val", "test", "test-0.1"]:
        num_queries = 0
        num_answers = 0
        stark_partition = partition
        indices = qa.split_indices[stark_partition].tolist()
        os.makedirs(f"{out}", exist_ok=True)
        with open(f"{out}/{partition}.queries", "w") as f_queries, open(f"{out}/{partition}.qrels", "w") as f_qrels:
            for idx in tqdm(indices):
                row = qa.data.iloc[idx]
                assert row.id == idx
                query = remove_irregularities(row.query)
                answer_ids = set(json.loads(row.answer_ids))
                if max_docs:
                    answer_ids = [answer_id for answer_id in answer_ids if answer_id < max_docs]
                if answer_ids:
                    print(f"{idx}\t{query}", file=f_queries)
                    num_queries += 1
                for answer_id in answer_ids:
                    print(trec.QRels(idx, answer_id, 1.0), file=f_qrels)
                    num_answers += 1
        print(f"Partition {partition} has {num_queries} queries and {num_answers} relevance judgements.")
        if num_queries == 0 and partition == "train":
            print(f"Partition {partition} has no queries. This will cause an error.")
            raise ValueError(f"Partition {partition} has no queries.")

if __name__ == "__main__":
    Fire(main)