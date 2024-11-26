from fire import Fire
from typing import List, Dict, Set

from mfar.data import trec
from mfar.data.typedef import Corpus


def emph(s: str) -> str:
    return f"\033[94;4;1m{s}\033[0m"


def main(*, data_path: str, partition: str, res1: str, res2: str, k: int):

    queries = Corpus.from_trec(f"{data_path}/{partition}.queries")
    with open(f"{data_path}/{partition}.qrels", 'r') as f_gold:
        gold = trec.QRels.from_text_io(f_gold)
    with open(res1) as f_pred:
        pred1 = trec.QRes.from_text_io(f_pred)
    with open(res2) as f_pred:
        pred2 = trec.QRes.from_text_io(f_pred)

    gold_ids: Dict[str, Set[str]] = {}
    for item in gold:
        if item.query_id not in gold_ids:
            gold_ids[item.query_id] = set()
        gold_ids[item.query_id].add(item.doc_id)

    pred_ids1: Dict[str, List[str]] = {}
    for item in pred1:
        if item.query_id not in pred_ids1:
            pred_ids1[item.query_id] = []
        pred_ids1[item.query_id].append(item.doc_id)

    pred_ids2: Dict[str, List[str]] = {}
    for item in pred2:
        if item.query_id not in pred_ids2:
            pred_ids2[item.query_id] = []
        pred_ids2[item.query_id].append(item.doc_id)

    for query_id in gold_ids:
        gold_set = gold_ids[query_id]
        pred_list1 = pred_ids1.get(query_id, [])[:k]
        pred_list2 = pred_ids2.get(query_id, [])[:k]
        pred_hit1 = len(gold_set & set(pred_list1)) > 0
        pred_hit2 = len(gold_set & set(pred_list2)) > 0
        if pred_hit1 != pred_hit2:
            print(f"Query {query_id} has different results")
            print(f"Query: {queries.get_text_by_key(query_id)}")
            print(f"Pred1: {', '.join(emph(x) if x in gold_set else x for x in pred_list1)}")
            print(f"Pred2: {', '.join(emph(x) if x in gold_set else x for x in pred_list2)}")
            print()


if __name__ == '__main__':
    Fire(main)
