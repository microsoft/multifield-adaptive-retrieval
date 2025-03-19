import multiprocessing
from fire import Fire
from mfar.data.index import BM25sSparseIndex
from mfar.data.typedef import FieldType
from mfar.data.schema import resolve_fields

from mfar.modeling.util import read_and_create_indices
import numpy as np
import tqdm
import gc

def precompute_score_for_field(index, all_candidate_docs, train_queries, output_path, field_key):
    index.set_safe_docs(all_candidate_docs)
    output_keys = []
    output_vals = []
    print(f"Working on {field_key}...")
    with multiprocessing.Pool(processes=64) as pool:
        raw_score_arrays = pool.map(index.get_scores_sparse, train_queries.values())
    for qid, score_array in tqdm.tqdm(zip(train_queries.keys(), raw_score_arrays),
                                      total=len(train_queries)):
        keys_dict = [(int(qid), int(doc_id)) for doc_id in score_array.keys()]
        vals_dict = [np.float16(score) for score in score_array.values()]
        output_keys.extend(keys_dict)
        output_vals.extend(vals_dict)

    output_keys_array = np.array(output_keys, dtype=np.int32)
    output_vals_array = np.array(output_vals, dtype=np.float16)

    np.save(f"{output_path}/{field_key}_keys_bm25.npy", output_keys_array)
    np.save(f"{output_path}/{field_key}_vals_bm25.npy", output_vals_array)
    print(f"{len(output_keys_array)} scores written to {output_path}/{field_key}.scores")

def main(
        data_path: str,
        dataset_name: str,
        corpus_path: str,
        output_path: str,
        index_path: str,
        fields_str: str="all_sparse,single_sparse"
):
    fields = resolve_fields(fields_str, dataset_name)
    # Ensure there are no dense fields
    if any(field.field_type == FieldType.DENSE for field in fields.values()):
        raise ValueError("Dense fields are not supported in this script.")

    _, _, indices_dict = read_and_create_indices(
        f"{corpus_path}/corpus",
        dataset_name,
        fields,
        None, # temp_dir is not used in this script
        None, # Since encoder is None, this will error if dense fields are passed in
    )

    train_queries = {}
    for partition in ["train"]:
        # Open queries and conver to dict from id to query
        with open(f"{data_path}/{partition}.queries", 'r') as f:
            for line in f:
                idx, query = line.strip().split("\t")
                train_queries[int(idx)] = query
        # Get positive docs
        pos_docs = set()
        with open(f"{data_path}/{partition}.qrels", 'r') as f:
            for line in f:
                qid, _, doc_id, _ = line.strip().split("\t")
                qid = int(qid)
                pos_docs.add(int(doc_id))

    print(f"Loaded {len(train_queries)} queries across train and val")
    print(f"Loaded {len(pos_docs)} positive docs across train and val")

    # For each query in the train partition, get the top-150 doc IDs
    negative_sampling_index = BM25sSparseIndex.load(f"{index_path}/single_sparse_sparse_index")
    retrieved_docs = negative_sampling_index.retrieve_batch(train_queries.values(), top_k=150)
    neg_candidate_docs = [int(doc_id) for result in retrieved_docs for doc_id, _ in result]
    print(f"Retrieved {len(neg_candidate_docs)} possible neg docs across train")
    neg_candidate_docs = set(neg_candidate_docs)
    print(f"Unique negative docs: {len(neg_candidate_docs)}")
    all_candidate_docs = neg_candidate_docs.union(pos_docs)
    print(f"Total candidate docs: {len(all_candidate_docs)}")

    # Write scores to files
    for field_key in list(indices_dict.keys()):
        index = indices_dict[field_key]
        precompute_score_for_field(index, all_candidate_docs, train_queries, output_path, field_key)
        del indices_dict[field_key]
        gc.collect()


if __name__ == "__main__":
    Fire(main)
