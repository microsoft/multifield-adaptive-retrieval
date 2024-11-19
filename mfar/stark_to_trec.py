from typing import List, Any, Dict
from tqdm import tqdm
import json
import numpy as np
import torch
from fire import Fire
from stark_qa import load_qa, load_skb
from stark_qa.retrieval import STaRKDataset
from stark_qa.skb import SKB

from util import remove_irregularities
import os

EDGE_FIELD_DICTS = {
    "amazon": {"also_buy": "title",
               "also_view": "title"},
    "mag": {"paper___cites___paper": "title",
            "author___writes___paper": "DisplayName",
            "paper___has_topic___field_of_study": "DisplayName",
            "author___affiliated_with___institution": "DisplayName"},
    "prime": {'ppi': 'name',
             'carrier': 'name',
             'enzyme': 'name',
             'target': 'name',
             'transporter': 'name',
             'contraindication': 'name',
             'indication': 'name',
             'off-label use': 'name',
             'synergistic interaction': 'name',
             'associated with': 'name',
             'parent-child': 'name',
             'phenotype absent': 'name',
             'phenotype present': 'name',
             'side effect': 'name',
             'interacts with': 'name',
             'linked to': 'name',
             'expression present': 'name',
             'expression absent': 'name'
    }
}

def main(*, dataset_name: str, out: str, max_docs: int = -1):
    if dataset_name not in EDGE_FIELD_DICTS:
        raise ValueError(f"Dataset name '{dataset_name}' is not recognized.")

    fields = EDGE_FIELD_DICTS[dataset_name]
    skb: SKB = load_skb(name=dataset_name, download_processed=True)
    indices: List[int] = skb.candidate_ids
    if max_docs != -1:
        indices = indices[:max_docs]
    os.makedirs(f"{out}", exist_ok=True)
    if dataset_name == "mag":
        skb_neighbor_cache = {}
    else:
        skb_neighbor_cache = None

    torch.set_num_threads(64)
    torch.set_num_interop_threads(64)
    def get_neighbor_subroutine(idx, t):
        if skb_neighbor_cache is not None:
            if (idx, t) not in skb_neighbor_cache:
                neighbors = skb.get_neighbor_nodes(idx, t)
                skb_neighbor_cache[(idx, t)] = neighbors
            else:
                neighbors = skb_neighbor_cache[(idx, t)]
        else:
            neighbors = skb.get_neighbor_nodes(idx, t)
        return neighbors

    output = []
    with open(f"{out}/corpus", "w") as f_corpus:
        for idx in tqdm(indices):
            node_info = skb.node_info[idx]
            edge_fields = {}
            for t in skb.edge_type_dict.values():
                if t in fields:
                    neighbors = get_neighbor_subroutine(idx, t)
                    # Case 1: if the person is an author, need to get the institutions (Mag)
                    if t == "author___writes___paper" and "author___affiliated_with___institution" in fields:
                        # We invoke the author___affiliated_with___institution edge type
                        neighbors_hop = [
                            get_neighbor_subroutine(n, "author___affiliated_with___institution")
                            for n in neighbors
                        ]
                        edge_fields[t] = neighbors
                        edge_fields["author___affiliated_with___institution"] = {
                            name: inst for name, inst in zip(neighbors, neighbors_hop)
                        }
                    # Case 2: the entire dataset (Prime)
                    elif dataset_name == "prime":
                        node_types = skb.node_types[neighbors]
                        neighbors = torch.tensor(neighbors)
                        for node_type in set(node_types.tolist()):
                            neighbors_t = neighbors[node_types == node_type]
                            try:
                                edge_fields[t][node_type] = neighbors_t.tolist()
                            except:
                                if t not in edge_fields:
                                    edge_fields[t] = {}
                                if node_type not in edge_fields[t]:
                                    edge_fields[t][node_type] = {}
                                edge_fields[t][node_type] = neighbors_t.tolist()
                    else:
                        edge_fields[t] = neighbors

            edge_info = {}
            for t in edge_fields:
                obj = edge_fields[t]
                if isinstance(obj, dict):
                    if t == "author___affiliated_with___institution":
                        edge_info[t] = {}
                        for key in obj:
                            for neighbor in obj[key]:
                                text = skb.node_info[key][fields["author___writes___paper"]]
                                if text != -1 and text != '-1':
                                    edge_info[t][text] = [skb.node_info[neighbor][fields[t]]]
                    else:
                        edge_info[t] = {
                            skb.node_type_dict[key]: [
                                skb.node_info[neighbor][fields[t]]
                                for neighbor in obj[key]
                            ] for key in obj
                        }
                else:
                    edge_info[t] = [
                        skb.node_info[neighbor][fields[t]]
                        for neighbor in obj if skb.node_info[neighbor][fields[t]] != -1 and \
                            skb.node_info[neighbor][fields[t]] != '-1'
                    ]

            doc = remove_irregularities(node_info | edge_info)
            output.append(f"{idx}\t{json.dumps(doc, ensure_ascii=False)}")

        f_corpus.write("\n".join(output))

    print(f"Corpus {dataset_name} has {len(indices[:max_docs])} documents.")

if __name__ == '__main__':
    Fire(main)
