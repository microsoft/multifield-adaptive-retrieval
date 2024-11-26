import argparse
from concurrent.futures import ThreadPoolExecutor

from mfar.data.typedef import Corpus
from mfar.data.schema import resolve_fields
from mfar.data import trec
from mfar.modeling.util import prepare_model
from mfar.data.format import format_documents
import numpy as np

from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--corpus", help="corpus name")
    parser.add_argument("--dataset_name", help="dataset name")
    parser.add_argument("--field_names", help="field names")
    args = parser.parse_args()

    # Open corpus file using Corpus.from_docs_dict:
    corpus = dict(trec.read_corpus(f"{args.corpus}/corpus"))
    documents = [(doc._id, doc.text) for doc in Corpus.from_docs_dict(corpus).docs]
    # Resolve the field names
    field_info = resolve_fields(args.field_names, args.dataset_name)

    # Print the field names
    print(field_info)
    num_docs = len(documents)
    print(num_docs)

    tokenizer, _, _ = prepare_model(
        "facebook/contriever-msmarco",
        normalize=False,
        with_decoder=False,
    )
    field_counter = {}

    field_names_list = list(field_info.keys())
    fields = list(field_info.values())
    formatted_docs = {}
    for field_name, field in field_info.items():
        formatted_docs[field_name] = format_documents(documents, field.name, field.dataset)

    for i in range(10):
        for field_name in field_info.keys():
            print(f"{field_name}\n\n")
            print(formatted_docs[field_name][i][1])

    all_field_doc_combinations = [(field, doc_id) for field in field_names_list for doc_id in range(num_docs)]
    print(len(all_field_doc_combinations))
    def count(arg_tuple):
        field_name = arg_tuple[0]
        doc_id = arg_tuple[1]
        doc = formatted_docs[field_name][doc_id]
        tokens = tokenizer.tokenize(doc)
        return len(tokens)

    def count_all_for_field(field_name):
        count_per_field = []
        for doc_id in tqdm(range(num_docs)):
            count_per_field.append(count((field_name, doc_id)))
        return count_per_field

    def count_batch(list_of_args):
        count_per_args = {}
        for item in tqdm(list_of_args):
            count_per_args[item] = count(item)
        return count_per_args

    # Can parallelize the tokenization/counting process by splitting the list of all_field_doc_combinations
    CHUNK_SIZE = 50000
    chunked_args = [all_field_doc_combinations[i:i + CHUNK_SIZE] for i in range(0, len(all_field_doc_combinations), CHUNK_SIZE)]

    with ThreadPoolExecutor() as executor:
        all_field_doc_lens = list(executor.map(count_batch, chunked_args)) # this is a list of dicts
    # Flatten this list of dicts into a single dict
    flattened_dict = {k: v for d in all_field_doc_lens for k, v in d.items()}

    for field_name in field_info.keys():
        field_counter[field_name] = [length for args, length in flattened_dict.items()
                                     if args[0] == field_name]

    # Make lots of space between above output and final output.
    for i in range(100):
        print("\n")

    for field_name, _ in field_info.items():
        relevant_percentiles = [
            max(field_counter[field_name]),
            int(np.ceil(np.percentile(field_counter[field_name], 99.9))),
            int(np.ceil(np.percentile(field_counter[field_name], 99))),
            int(np.ceil(np.percentile(field_counter[field_name], 95))),
            int(np.ceil(np.percentile(field_counter[field_name], 90))),
            int(np.ceil(np.percentile(field_counter[field_name], 75))),
            int(np.ceil(np.percentile(field_counter[field_name], 50))),
        ]
        print(f"{field_name}," + ",".join(map(str, relevant_percentiles)))
    print(num_docs)