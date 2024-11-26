from fire import Fire
from typing import List, Dict, Set

from mfar.data import trec
from mfar.data.typedef import Corpus


def main(*, data_path: str, partition: str, qres_path: str, k: int):

    corpus = Corpus.from_trec(f"{data_path}/corpus")
    queries = Corpus.from_trec(f"{data_path}/{partition}.queries")

    with open(f"{data_path}/{partition}.qrels", 'r') as f_gold:
        gold = trec.QRels.from_text_io(f_gold)
    with open(qres_path) as f_pred:
        pred = trec.QRes.from_text_io(f_pred)

    gold_ids: Dict[str, Set[str]] = {}
    for item in gold:
        if item.query_id not in gold_ids:
            gold_ids[item.query_id] = set()
        gold_ids[item.query_id].add(item.doc_id)

    pred_ids: Dict[str, List[str]] = {}
    for item in pred:
        if item.query_id not in pred_ids:
            pred_ids[item.query_id] = []
        pred_ids[item.query_id].append(item.doc_id)

    for query_id in gold_ids:
        if query_id not in pred_ids:
            print(f"Query {query_id} not found in predictions")
            continue
        gold_set = gold_ids[query_id]
        pred_list = pred_ids[query_id]
        if len(gold_set & set(pred_list[:k])) == 0 and len(gold_set & set(pred_list[k:])) > 0:
            gold_ranks = [(i, doc_id) for i, doc_id in enumerate(pred_list) if doc_id in gold_set]
            gold_ranks_sorted = sorted(gold_ranks, key=lambda t: t[0])

            print(f"QUERY: {queries.get_text_by_key(query_id)}")
            print(f"Top {k} retrieved: {pred_list[:k]}")
            print(f"Relevant documents: {gold_ranks_sorted}")
            print()


if __name__ == '__main__':
    Fire(main)
