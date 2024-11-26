from datetime import timedelta
from typing import *

import os
import time
import torch
from fire import Fire
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from mfar.modeling.util import prepare_model
from mfar.data.util import MLFlowLoggerWrapper
from mfar.modeling.contrastive import RetrievalDataModule, \
    RetrievalTrainingModule

from mfar.data.schema import resolve_fields
import json

def main(*,
         dataset_name: str,
         lexical_index: str,
         out: str,
         temp_dir: str,
         partition: str = "val",
         data: Optional[str]=None,
         queries: Optional[str]=None,
         corpus: Optional[str]=None,
         additional_partition: Optional[str] = None,
         model_name: str = "facebook/contriever-msmarco",
         normalize: bool = False,
         temperature: float = 0.05,
         negative_sampling_params: Tuple[int, int, int] = (100, 50, 1),
         encoder_lr: float = 1e-4,
         weights_lr: Optional[float] = None,
         regularizer: float = 0.0,
         train_batch_size: int = 16,
         dev_batch_size: int = 64,
         train_max_length: int = 512,
         dev_max_length: int = 512,
         max_epochs: int = 50,
         patience: int = 10,
         seed: int = 0xdeadbeef,
         precision: str = "16-mixed",
         num_gpus: int = -1,
         dev_by_iter: bool = False,
         logger: Optional[str] = None,
         freeze_encoder: bool = False,
         wandb_name: str = None,
         wandb_dir: str = None,
         experiment_name: str = None,
         field_names: List = None,
         trec_val_freq: int = 0,
         query_cond: bool = True,
         prefix: bool = False,
         run_one_iteration = False,
         use_batchnorm: bool = False,
         ):

    torch.set_float32_matmul_precision("high")
    pl.seed_everything(seed)

    field_info = resolve_fields(field_names, dataset_name)
    field_info_string = json.dumps({k: v.__dict__() for k, v in field_info.items()})
    if logger == "wandb":
        logger = WandbLogger(project=wandb_name, group=experiment_name, log_model=False, save_dir=wandb_dir)
    elif logger == "mlflow":
        logger = MLFlowLoggerWrapper(
            experiment_name=os.getenv("AZURE_EXPERIMENT_NAME"),
            run_id=os.getenv("AZUREML_RUN_ID"),
            tracking_uri=os.getenv("MLFLOW_TRACKING_URI"),
        )
    elif logger == "mlflow_local":
        logger = MLFlowLoggerWrapper(tracking_uri=os.getenv("MLFLOW_LOCAL_PATH"))
    else:
        logger = None
    val_frequency = f"Every {trec_val_freq} (starting at {trec_val_freq-1})" if trec_val_freq > 0 else "Proxy only"

    dev_by_iter = (dataset_name == "amazon")

    # Input can either be a single `data` directory or separate `queries` and `corpus` directories.
    if data:
        queries = data
        corpus = data
    os.makedirs(out, exist_ok=True)

    print(
        f"""Starting training with the following experimental settings:
        - Model: {model_name}
        - Queries: {queries}
        - Corpus: {corpus}
        - Dataset name: {dataset_name}
        - Fields: {field_info_string}
        - Prefix: {prefix}
        - Query truncation: {train_max_length}
        - Encoder LR: {encoder_lr}
        - Weights LR: {weights_lr}
        - TREC Validation Frequency: {val_frequency}
        - Test Set: {additional_partition}
        - Batch norm: {use_batchnorm}
        - Current time: {time.strftime("%Y-%m-%d %H:%M:%S")}
        """
    )

    tokenizer, encoder, _ = prepare_model(
        model_name,
        normalize=normalize,
        with_decoder=False,
    )
    data_module = RetrievalDataModule(
        tokenizer=tokenizer,
        queries_path=f"{queries}",
        corpus_path=f"{corpus}",
        temp_path=temp_dir,
        dev_partition=partition,
        additional_partition=additional_partition,
        lexical_index=lexical_index,
        negative_sampling_params=negative_sampling_params,
        train_batch_size=train_batch_size,
        dev_batch_size=dev_batch_size,
        train_max_length=train_max_length,
        dev_max_length=dev_max_length,
        dataset_name=dataset_name,
        field_info=field_info,
        prefix=prefix,
        trec_val_freq=trec_val_freq,
    )
    module = RetrievalTrainingModule(
        encoder=encoder,
        model_id=model_name,
        decoder=None,
        contrastive_temp=temperature,
        dev_qrels_path=f"{queries}/{partition}.qrels",
        additional_qrels_path=f"{queries}/{additional_partition}.qrels" if additional_partition else None,
        corpus_path=f"{corpus}/corpus",
        dataset_name=dataset_name,
        encoder_learning_rate=encoder_lr,
        weights_learning_rate=weights_lr,
        weight_decay=regularizer,
        dev_batch_size=dev_batch_size,
        temp_dir=temp_dir,
        out_dir=out,
        field_info=field_info,
        trec_val_freq=trec_val_freq,
        freeze_encoder=freeze_encoder,
        query_cond=query_cond,
        prefix=prefix,
        use_batchnorm=use_batchnorm,
    )
    if logger is not None:
        logger.log_hyperparams({
            "queries_dir": queries,
            "corpus_dir": corpus,
            "lexical_index": lexical_index,
            "output_dir": out,
            "vector_dir": temp_dir,
            "partition": partition,
            "model_name": model_name,
            "normalize": normalize,
            "temperature": temperature,
            "negative_sampling_params": negative_sampling_params,
            "encoder_lr": encoder_lr,
            "weights_lr": weights_lr,
            "regularizer": regularizer,
            "train_batch_size": train_batch_size,
            "dev_batch_size": dev_batch_size,
            "train_max_length": train_max_length,
            "dev_max_length": dev_max_length,
            "max_epochs": max_epochs,
            "patience": patience,
            "seed": seed,
            "precision": precision,
            "num_gpus": num_gpus,
            "dev_by_iter": dev_by_iter,
            "freeze_encoder": freeze_encoder,
            "wandb_name": wandb_name,
            "experiment_name": experiment_name,
            "field_info": field_info_string,
            "prefix": prefix,
            "trec_val_freq": trec_val_freq,
            "query_cond": query_cond,
            "run_one_iteration": run_one_iteration,
        })

    # This part of the code can be modified more, e.g. by specifying what to monitor as a flag.
    monitor = "valid_loss"
    mode = "min"
    filename = '{epoch}-{valid_loss:.3f}'

    trainer = pl.Trainer(
        callbacks=[
            EarlyStopping(
                monitor=monitor,
                patience=patience,
                mode=mode,
                check_on_train_epoch_end=False,
                strict=False,
            ),
            ModelCheckpoint(
                dirpath=out,
                monitor=monitor,
                filename=filename,
                save_last=True,
                save_top_k=100,
                save_weights_only=True,
                mode=mode,
            ),
        ],
        max_epochs=max_epochs,
        logger=logger,
        accelerator="gpu",
        devices=num_gpus,
        strategy=DDPStrategy(find_unused_parameters=True, timeout=timedelta(hours=12)),
        precision=precision,
        use_distributed_sampler=False,
        reload_dataloaders_every_n_epochs=1,
        log_every_n_steps=10,
        val_check_interval=0.2 if dev_by_iter else None,
        check_val_every_n_epoch=1, # None if dev_by_iter else 1,
        limit_train_batches=1 if run_one_iteration else None,
        limit_val_batches=1 if run_one_iteration else None,
    )

    print(f"Starting training: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    trainer.fit(module, data_module)
    best = trainer.callbacks[-1].best_model_path
    trainer.test(module, data_module, ckpt_path="best", verbose=True)

    with open(f"{out}/best.txt", "w") as f:
        f.write(str(best))

if __name__ == "__main__":
    Fire(main)
