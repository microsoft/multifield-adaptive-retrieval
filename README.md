# Multi-Field Adaptive Retrieval

This is the repository for Multi-field Adaptive Retrieval (mFAR).

## Getting Started

Create a python 3.10 environment:

`conda create -n mfar python=3.10`

Next, install the poetry environment in this main directory.

```
# Ideally nothing would change after poetry lock, but sometimes libraries update.
poetry lock
poetry install
```

## Data Processing

Use the official library (`stark-qa`) to download the data (already included if you use installed the environment).

Parameter definitions:
```
out: directory where data should be stored
dataset_name: {"mag", "prime", "amazon"}
max_docs: integer to download or create a truncated version of the corpus or dataset, e.g. 2000 is reasonably small.
```

In these examples we will use `$CORPUS_DIR`, `$QUERY_DIR`, and `$INDEX_DIR` but in practice they can all be the same directory since there are no overwritten files.

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
logger: Optional string of "wandb", "mlflow", "mlflow_local". Note that "mlflow_local" will require setting the MLFLOW_LOCAL_PATH environment variable. `mlflow` is intended for use with AzureML.
```


## Analysis and Other Commands

Here is one example of masking out zeros per field. `debug=True` only runs model load/test once and does not actually mask any fields. `field_names` must be the same field names as what the was trained with, otherwise we will have a dimension mismatch and mask the incorrect fields.

```
python -m mfar.commands.mask_fields --corpus $CORPUS_DIR --queries $QUERY_DIR --dataset_name prime --lexical-index $INDEX_DIR --temp-dir $TEMP_DIR --out $ANALYSIS_OUT_DIR --field_names "all_dense" --logger mlflow_local --additional_partition test --checkpoint_dir $MODEL_OUT_DIR --debug True
```

## Contributing

See CONTRIBUTING.md

## Citation

```
@misc{li2024multifieldadaptiveretrieval,
      title={Multi-Field Adaptive Retrieval},
      author={Millicent Li and Tongfei Chen and Benjamin Van Durme and Patrick Xia},
      year={2024},
      eprint={2410.20056},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2410.20056},
}
```

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.