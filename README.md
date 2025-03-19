# Multi-Field Adaptive Retrieval

This is the code repository for ICLR 2025 paper [Multi-field Adaptive Retrieval](https://openreview.net/forum?id=3PDklqqqfN) (mFAR).

## Getting Started

Create a python 3.10 environment:

`conda create -n mfar python=3.10`

And install poetry 1.8.4 (future versions might also work)

`pip install poetry==1.8.4`

Next, install the poetry environment in this main directory.

```
# Ideally nothing would change after poetry lock, but sometimes libraries update.
poetry lock --no-update
poetry install
```

To run evaluation, we will need `trec_eval`. That can be installed from [https://github.com/usnistgov/trec_eval](https://github.com/usnistgov/trec_eval) and built with

```
cd trec_eval
make
make install
```

We also include a Dockerfile that might be helpful for building the environment (e.g. if submitting to cluster compute).

## Data Processing

Use the official library (`stark-qa`) to download the data (already included if you use installed the environment).

Parameter definitions:
```
out: directory where data should be stored
dataset_name: {"mag", "prime", "amazon"}
max_docs: integer to download or create a truncated version of the corpus or dataset, e.g. 2000 is reasonably small.
```

In these examples we will use `$CORPUS_DIR`, `$QUERY_DIR`, and `$INDEX_DIR` but in practice they can all be the same directory since there are no overwritten files. In these examples we do not include `--max_docs <number>` but for initial set-up, it is recommended to quickly ensure all the steps run properly.

To download and format the corpus:
```
python -m mfar.commands.stark.stark_to_trec --out $CORPUS_DIR --dataset_name $DATASET_NAME
```

To download the queries
```
python -m mfar.commands.stark.download_queries --out $QUERY_DIR --dataset_name $DATASET_NAME
```

We need to create a lexical index for negative sampling during training or if we want to use "sparse" fields. `data_path` is the location of the corpus file.

```
python -m mfar.commands.create_bm25s_index --data_path $CORPUS_DIR --dataset_name $DATASET_NAME --output_path $INDEX_DIR
```

(Experimental) Finally, to improve computation efficiency, we can pre-compute all of the BM25 scores that we would need during training. Note these files can get large and loading into memory is slow and we found it wasn't worth doing in practice.

```
python -m mfar.commands.precompute_bm25s_scores --data_path $QUERY_DIR --dataset_name  $DATASET_NAME --corpus_path $CORPUS_DIR --index_dir $INDEX_DIR --output_path $SPARSE_SCORES_DIR
```

## Training models

Here is one example of training. There are many defaults (encoder, hyperparameters) that can be changed so look at `train.py` to see the other options. `$TEMP_DIR` can be deleted/ovewritten after each run, it is used for temporarily storing vectors.

```
python -m mfar.commands.train --corpus $CORPUS_DIR --queries $QUERY_DIR --lexical-index $INDEX_DIR --temp-dir $TEMP_DIR --out $MODEL_OUT_DIR --encoder_lr 1e-5 --weights_lr 1e-1 --train-batch-size 12 --field_names "all_dense" --trec_val_freq 1 --dataset_name prime --logger mlflow_local
```

In the above command:
```
encoder_lr: learning rate of the underlying text encoder
weights_lr: learning rate for the query-conditioned mixture weights/field embeddings
train_batch_size: modify based on dataset/disk. For 8xA100s, {12, 24, 12} work for {prime, mag, amazon}
field_names: comma-separated list of strings of the form f"{SCORER}_{FIELD}" where SCORER can be "all" or "single" and FIELD is the name of a field. See schema.py for more examples.
trec_val_freq: how often to run the full treq eval (could be slow)
dataset_name: "amazon", "mag", "prime"
sparse_scores_dir: optional, $SPARSE_SCORES_DIR if they were precomputed in the previous step (otherwise they will be computed and cached from scratch during the forward pass)
logger: Optional string of "wandb", "mlflow", "mlflow_local". Note that "mlflow_local" will require setting the MLFLOW_LOCAL_PATH environment variable. `mlflow` is intended for use with AzureML.
```


## Inference or Analysis

We can also reload the model for evaluation. By default, evaluation includes the field masking analysis from the paper. However, if we are simply interested in running eval, we can set `--debug True`. If we wish to run the full analysis, leave this argument out.

```
python -m mfar.commands.mask_fields --corpus $CORPUS_DIR --queries $QUERY_DIR --dataset_name prime --lexical-index $INDEX_DIR --temp-dir $TEMP_DIR --out $ANALYSIS_OUT_DIR --field_names "all_dense" --logger mlflow_local --additional_partition test --checkpoint_dir $MODEL_OUT_DIR --debug True
```

The analysis scripts in `/scripts` may no longer be compatible with the final version of the code.

## Bring your own dataset

_These instructions are not thoroughly tested._

To train a model on an arbitrary dataset, there are the changes that need to be made. Note that in the paper, each trained model is "tied" to the data schema that was used to train the model (i.e. a MAG model won't work on Amazon). This does not necessarily need to be the case, but the training process would need some modifications.

Here, we describe an example of training a model on an entirely new dataset. This involves 3 steps:

1. Ensure the original data is in the correct (TREC) format.
2. Add the data schema to `schema.py`
3. Add a `format` function in `format.py`

One example is already included with the `WTB` (books) dataset. We did not run many experiments with it but it should give an idea of what needs to change.

### Input format

The dataset should fit the TREC format. This includes a `corpus` file which is a tsv with an integer `id` column and a `json` document column (do not include column headers):

```
doc_id    {"field_1": ..., "field2": ..."}
```

The queries should consist of `{train, test, val}.queries` which is a tsv with integer `id` column and `string` query column (do not include column headers):

```
query_id    Is there a new men's t-shirt on Amazon that features sleeves sewn directly into the shoulder seam?
```

And there should be a labels file, `{train, test, val}.qrels` which consists of 4 columns, `query_id`, unused, `doc_id`, unused, and `relevance`. Relevance is also unused, and is assumed to be 1.0.

### Updating the available schemas

First, determine what fields are available in the dataset, and add them to `data/schema.py`, along with the (expected) max length of the field - for Contriever, this cannot be greater than 512. Also add them to `FIELDS_DICT`.

### Adding formatting code

Within the code, each field of the document is formatted using `format_documents(document, field_name, dataset_name)` in `data/format.py`. For individual fields (like int or string), this simply returns a string or integer. For more complex fields or special cases (dependent on the type or dataset), additional formatting may apply.

In addition, the "full document" view of a document is formatted using `format_stark(document, dataset_name)`. After adding a dataset, we recommend also writing a custom `format_<your dataset>` function and register it as a case under `format_stark()`.

## Possible issues

Here are some known possible issues.

_CPU out of memory_

If the corpus is too large relative to the device's RAM, the `maxsize` of the `lru_cache` in `data/index.py` may need to be lowered. Note there are two; lowering the one for `get_scores` has a bigger effect.

_GPU out of memory_

One option is to the batch sizes. By default we use 24 for MAG, 12 for Prime, and 12 for Amazon on 8xA100s. Alternatively, we can reduce the sum of the lengths of the fields. For example, MAG is set to 512+512+512+64+64 = 1664. If we halved each of these values, we may be able to double batch size (not exactly - it depends on which fields are being run).

We do not have support for gradient checkpointing in this codebase, but it might be easy to do with Lightning.

## Contributing

See CONTRIBUTING.md

## Citation

```
@inproceedings{
      li2025multifield,
      title={Multi-Field Adaptive Retrieval},
      author={Millicent Li and Tongfei Chen and Benjamin Van Durme and Patrick Xia},
      booktitle={The Thirteenth International Conference on Learning Representations},
      year={2025},
      url={https://openreview.net/forum?id=3PDklqqqfN}
}
```

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.