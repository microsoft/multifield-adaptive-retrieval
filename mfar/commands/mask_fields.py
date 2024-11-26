from datetime import timedelta
from typing import *

import os
import time
import torch
from fire import Fire
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy

from mfar.modeling.util import prepare_model
from mfar.data.util import MLFlowLoggerWrapper
from mfar.modeling.contrastive import RetrievalDataModule, \
    RetrievalTrainingModule

from mfar.data.schema import resolve_fields
from mfar.data.typedef import FieldType

def main(*,
         dataset_name: str,
         lexical_index: str,
         out: str,
         temp_dir: str,
         data: Optional[str]=None,
         queries: Optional[str]=None,
         corpus: Optional[str]=None,
         partition: str = "val",
         additional_partition: Optional[str] = None,
         model_name: str = "facebook/contriever-msmarco",
         normalize: bool = False,
         negative_sampling_params: Tuple[int, int, int] = (100, 50, 1),
         train_batch_size: int = 16,
         dev_batch_size: int = 64,
         train_max_length: int = 512,
         dev_max_length: int = 512,
         max_epochs: int = 50,
         seed: int = 0xdeadbeef,
         precision: str = "16-mixed",
         num_gpus: int = -1,
         logger: Optional[str] = None,
         wandb_name: str = None,
         wandb_dir: str = None,
         experiment_name: str = None,
         field_names: List = None,
         trec_val_freq: int = 0,
         prefix: bool = False,
         checkpoint_dir: Optional[str] = None,
         debug: bool = False,
         ):

    torch.set_float32_matmul_precision("high")
    pl.seed_everything(seed)
    if data:
        queries = data
        corpus = data

    field_info = resolve_fields(field_names, dataset_name)
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

    tokenizer, encoder, _ = prepare_model(
        model_name,
        normalize=normalize,
        with_decoder=False,
    )
    data_module = RetrievalDataModule(
        tokenizer=tokenizer,
        queries_path=queries,
        corpus_path=corpus,
        dataset_name=dataset_name,
        temp_path=temp_dir,
        dev_partition=partition,
        additional_partition=additional_partition,
        lexical_index=lexical_index,
        negative_sampling_params=negative_sampling_params,
        train_batch_size=train_batch_size,
        dev_batch_size=dev_batch_size,
        train_max_length=train_max_length,
        dev_max_length=dev_max_length,
        field_info=field_info,
        prefix=prefix,
        trec_val_freq=trec_val_freq,
    )

    best_ckpt = open(f"{checkpoint_dir}/best.txt", "r")
    checkpoint_suffix = best_ckpt.read().strip().split("/")[-1]
    checkpoint_path = f"{checkpoint_dir}/{checkpoint_suffix}"
    print(f"PATH IS: {checkpoint_path}")
    module = RetrievalTrainingModule.load_from_checkpoint(
        checkpoint_path,
        corpus_path=f"{corpus}/corpus",
        dev_qrels_path=f"{queries}/{partition}.qrels",
        additional_qrels_path=f"{queries}/{additional_partition}.qrels" if additional_partition else None,
        encoder=encoder,
        field_info=field_info,
        temp_dir=temp_dir,
        out_dir=out,
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        logger=logger,
        accelerator="gpu",
        devices=num_gpus,
        strategy=DDPStrategy(find_unused_parameters=True, timeout=timedelta(hours=12)),
        precision=precision,
        use_distributed_sampler=False,
    )

    print(f"Starting re-testing of {checkpoint_path}: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    import logging
    log = logging.getLogger("pytorch_lightning")
    log.propagate = False
    log.setLevel(logging.ERROR)
    import warnings
    warnings.filterwarnings("ignore", ".*does not have many workers.*")

    print("Baseline Evaluation")
    trainer.test(module, data_module, verbose=False)



    if not debug:
        for idx in range(len(field_info)):
            module.mask_field([idx])
            trainer.test(module, data_module, verbose=False)

        sparse_idx = [idx for idx, field in enumerate(field_info.values()) if field.field_type == FieldType.SPARSE]
        if sparse_idx:
            module.mask_field(sparse_idx)
            trainer.test(module, data_module, verbose=False)
        else:
            print("No sparse fields")

        dense_idx = [idx for idx, field in enumerate(field_info.values()) if field.field_type == FieldType.DENSE]
        if dense_idx:
            module.mask_field(dense_idx)
            trainer.test(module, data_module, verbose=False)
        else:
            print("No dense fields")

        all_field_names = set(list([field.name for field in field_info.values()]))
        for name in all_field_names:
            masked_idx = [idx for idx, field in enumerate(field_info.values()) if field.name == name]
            module.mask_field(masked_idx)
            trainer.test(module, data_module, verbose=False)


if __name__ == "__main__":
    Fire(main)
