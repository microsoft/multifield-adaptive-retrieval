import pickle
from functools import reduce
from typing import Tuple, List, Optional, Union, TextIO, Dict
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import json
import torch
import torch.distributed
import transformers
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader, DistributedSampler, Dataset

from transformers import PreTrainedTokenizer, PreTrainedModel
import pytorch_lightning as pl

from mfar.data import trec
from mfar.data.index import Index, BM25sSparseIndex, candidate_encoding_stream
from mfar.data.format import format_documents
from mfar.data.negative_sampler import IndexNegativeSampler
from mfar.data.typedef import Corpus, Field, FieldType
from mfar.data.dataset import (
    ContrastiveTrainingDataset, QueryDataset, Kind,
    InstanceBatch, DecomposedInstanceBatch, any_collate
)

from mfar.modeling.losses import HybridContrastiveLoss
from mfar.modeling.weighting import LinearWeights

class RetrievalDataModule(pl.LightningDataModule):

    def __init__(self,
                 tokenizer: PreTrainedTokenizer,
                 queries_path: str,
                 corpus: List[Tuple[str, str]],
                 temp_path: str,
                 dev_partition: str,
                 additional_partition: Optional[str],
                 lexical_index: str,
                 negative_sampling_params: Tuple[int, int, int],
                 dataset_name: str,
                 train_batch_size: int = 64,
                 dev_batch_size: int = 64,
                 query_max_length: int = 64,
                 train_max_length: int = 384,
                 dev_max_length: int = 512,
                 dim: int = 768,
                 field_info: Dict[str, Field] = None,
                 indices_dict: Dict[str, Index] = None,
                 prefix: bool = False,
                 trec_val_freq: int = 0,
                 ):
        super().__init__()
        self.tokenizer = tokenizer
        self.queries_path = queries_path
        self.corpus = corpus
        self.temp_path = temp_path
        self.dev_partition = dev_partition
        self.additional_partition = additional_partition
        self.train_batch_size = train_batch_size
        self.dev_batch_size = dev_batch_size
        self.query_max_length = query_max_length
        self.train_max_length = train_max_length
        self.dev_max_length = dev_max_length

        self._log_hyperparams = True

        corpus = dict(corpus)
        self.documents = Corpus.from_docs_dict(corpus, dataset_name)
        self.negative_samplers = [IndexNegativeSampler(
            index=BM25sSparseIndex.load(f"{lexical_index}/single_sparse_sparse_index"),
            documents=corpus,
            n_retrieve=negative_sampling_params[0],
            n_bottom=negative_sampling_params[1],
            n_sample=negative_sampling_params[2],
        )]
        self.train_qrels = trec.QRels.from_text_io(open(f"{self.queries_path}/train.qrels"))
        self.train: Optional[Dataset] = None
        self.dev_qrels = trec.QRels.from_text_io(open(f"{self.queries_path}/val.qrels"))
        self.dev_queries: Optional[QueryDataset] = None

        self.prepare_data_per_node = True  # fdsp
        self.dim = dim
        self.field_info = field_info
        # Optimization for dataloading
        self.field_types = {field.field_type for field in field_info.values()}
        self.indices_dict = indices_dict # This dict needs to be mutated by the training module during eval

        self.prefix = prefix
        self.trec_val_freq = trec_val_freq
        self.dev_queries_dict = dict(trec.read_corpus(f"{self.queries_path}/{self.dev_partition}.queries"))
        if self.additional_partition:
            self.additional_queries_dict = [dict(trec.read_corpus(f"{self.queries_path}/{self.additional_partition}.queries"))]
        else:
            self.additional_queries_dict = []

    @property
    def corpus_size(self):
        return len(self.documents)

    @property
    def train_qrels_size(self):
        return len(self.train_qrels) if self.train_qrels is not None else 0

    def setup(self, stage: str) -> None:
        self.dev_queries = QueryDataset(
            tokenizer=self.tokenizer,
            queries=self.dev_queries_dict,
            max_length=self.query_max_length,
            field_types=self.field_types,
        )
        self.additional_queries = [
            QueryDataset(
                tokenizer=self.tokenizer,
                queries=queries_dict,
                max_length=self.query_max_length,
                field_types=self.field_types,
            )
            for queries_dict in self.additional_queries_dict
        ]

    def train_dataloader(self) -> DataLoader:
        supervised = ContrastiveTrainingDataset(
            tokenizer=self.tokenizer,
            queries=dict(trec.read_corpus(f"{self.queries_path}/train.queries")),
            documents=self.documents,
            qrels=self.train_qrels,
            negative_sampler=self.negative_samplers[0],
            max_length=self.train_max_length,
            field_info=self.field_info,
            field_types=self.field_types,
            indices_dict=self.indices_dict,
            prefix=self.prefix,
        )
        self.train = supervised
        sampler = DistributedSampler(self.train)
        batch_sampler = None
        batch_size = self.train_batch_size
        return DataLoader(
            dataset=self.train,
            batch_size=batch_size,
            collate_fn=lambda instances: any_collate(dataset=self.train, instances=instances),
            sampler=sampler,
            batch_sampler=batch_sampler,
            # num_workers=2,
            # prefetch_factor=4,
        )

    def val_dataloader(self) -> List[DataLoader]:
        data_loaders = []
        self.dev = ContrastiveTrainingDataset(
            tokenizer=self.tokenizer,
            queries=dict(trec.read_corpus(f"{self.queries_path}/val.queries")),
            documents=self.documents,
            qrels=self.dev_qrels,
            negative_sampler=self.negative_samplers[0],
            max_length=self.dev_max_length,
            field_info=self.field_info,
            field_types=self.field_types,
            indices_dict=self.indices_dict,
            random_chunk=True,
        )

        sampler = DistributedSampler(self.dev)
        batch_sampler = None
        batch_size = self.dev_batch_size
        data_loaders.append(DataLoader(
            dataset=self.dev,
            batch_size=batch_size,
            collate_fn=lambda instances: any_collate(dataset=self.dev, instances=instances),
            sampler=sampler,
            batch_sampler=batch_sampler,
            # num_workers=2,
            # prefetch_factor=4,
        ))

        # Trec eval
        if self.trec_val_freq > 0 and (self.trainer.current_epoch + 1) % self.trec_val_freq == 0:
            data_loaders.append(DataLoader(
                dataset=self.dev_queries,
                batch_size=self.dev_batch_size,
                collate_fn=self.dev_queries.collate,
                sampler=DistributedSampler(self.dev_queries),
                # num_workers=2,
                # prefetch_factor=4,
            ))
        else:
            data_loaders.append(DataLoader(dataset=[]))

        return data_loaders


    def test_dataloader(self) -> DataLoader:
        data_loaders = []
        data_loaders.append(DataLoader(
                dataset=self.dev_queries,
                batch_size=self.dev_batch_size,
                collate_fn=self.dev_queries.collate,
                sampler=DistributedSampler(self.dev_queries),
        ))
        for queries in self.additional_queries:
            data_loaders.append(DataLoader(
                dataset=queries,
                batch_size=self.dev_batch_size,
                collate_fn=queries.collate,
                sampler=DistributedSampler(queries),
            ))

        return data_loaders

class RetrievalTrainingModule(pl.LightningModule):
    def __init__(
            self,
            encoder: SentenceTransformer,
            model_id: str,
            decoder: Optional[PreTrainedModel],
            corpus_path: str,
            corpus: List[Tuple[str, str]],
            dataset_name: str,
            dev_qrels_path: str,
            out_dir: str,
            sparse_scores: Optional[Dict] = None,
            contrastive_temp: float = 0.01,
            encoder_learning_rate: float = 1e-5,
            weights_learning_rate: Optional[float] = None,
            weight_decay: float = 0.0,
            dev_batch_size: int = 32,
            field_info: Dict = None,
            indices_dict: Dict = None,
            vectors_dict: Dict = None,
            trec_val_freq: int = 0,
            freeze_encoder: bool = False,
            query_cond: bool = True,
            prefix: bool = False,
            additional_qrels_path: Optional[str] = None,
            use_batchnorm: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['encoder', 'corpus', 'indices_dict', 'vectors_dict', 'precomputed_sparse_scores'])
        self.encoder = encoder
        self.model_id = model_id
        self.decoder = decoder
        self.dataset_name = dataset_name
        self.corpus = corpus
        self.corpus_path = corpus_path # used for reloading models
        self.numeric_ids_to_keys = [x[0] for x in self.corpus]
        self.keys_to_numeric_ids = {k: i for i, k in enumerate(self.numeric_ids_to_keys)}

        if field_info == None:
            raise NotImplementedError("No fields passed in!")
        self.indices_dict = indices_dict
        self.vectors_dict = vectors_dict

        self.automatic_optimization = False # We have two optimizers
        self.encoder_learning_rate = encoder_learning_rate
        self.weights_learning_rate = weights_learning_rate
        self.weight_decay = weight_decay
        self.dev_batch_size = dev_batch_size
        self.dev_qrels_path = dev_qrels_path
        self.additional_qrels_path = additional_qrels_path
        self.out_dir = out_dir

        self.n_docs = len(self.corpus)
        self.best_score = 0.0
        self.field_info = field_info
        self.trec_val_freq = trec_val_freq
        self.query_cond = query_cond
        self.prefix = prefix
        self.mask = torch.ones([len(self.field_info), 1], device=self.device)
        self.masked_fields_string = ""

        if self.weights_learning_rate is None:
            raise ValueError(f"Need to specify a learning weight for the weights for {self.index_method}!")


        num_fields = len(field_info)
        if query_cond:
            self.mixture_of_fields_layer = LinearWeights(
                self.encoder.get_sentence_embedding_dimension(),
                num_fields,
                query_cond=True
            )
        else:
            self.mixture_of_fields_layer = LinearWeights(num_fields, 1)
        sparse_indices_dict = {field.key: self.indices_dict[field.key] for field in self.field_info.values() if field.field_type == FieldType.SPARSE}
        self.hybrid_contrastive_loss_fn = HybridContrastiveLoss(
            temperature=contrastive_temp,
            mixture_of_fields_layer=self.mixture_of_fields_layer,
            sparse_indices_dict=sparse_indices_dict,
            num_fields = num_fields,
            use_batchnorm = use_batchnorm,
        )

        self.qres_output: Optional[TextIO] = None
        self.additional_qres_output: Optional[TextIO] = None

        # Unpack precomputed sparse scores to dicts
        if sparse_scores:
            self.precomputed_sparse_scores = sparse_scores
        else:
            print(f"Did not find precomputed sparse scores (e.g. if loading from checkpoint)")
            self.precomputed_sparse_scores = {field_key: {} for field_key, field in self.field_info.items() if field.field_type == FieldType.SPARSE}

    def configure_optimizers(self) -> torch.optim.Optimizer:
        model_params = {
            param_name: weight for param_name, weight in self.named_parameters()
            if "encoder" in param_name and weight.requires_grad
        }
        linear_params = {
            param_name: weight for param_name, weight in self.named_parameters()
            if ("mixture_of_fields_layer" in param_name or "bn" in param_name) and weight.requires_grad
        }
        # Hopefully this is all the parameters
        unused_params = {
            param_name: weight for param_name, weight in self.named_parameters()
            if "encoder" not in param_name and "mixture_of_fields_layer" not in param_name and "bn" not in param_name
        }
        print("Untouched params:", unused_params)

        if "t5" in self.model_id:
            if len(linear_params) == 0:
                return transformers.Adafactor(  # used to train T5 to conserve GPU memory
                    self.parameters(),
                    lr=self.encoder_learning_rate,
                    relative_step=False,
                    scale_parameter=False,
                    warmup_init=False,
                    decay_rate=-0.8,
                    clip_threshold=1.0,
                )
            else:
                encoder_optim = transformers.Adafactor(
                    model_params.values(),
                    lr=self.encoder_learning_rate,
                    relative_step=False,
                    scale_parameter=False,
                    warmup_init=False,
                    decay_rate=-0.8,
                    clip_threshold=1.0,
                )

                linear_optim = torch.optim.AdamW(
                    linear_params.values(),
                    lr=self.weights_learning_rate,
                )

                return [encoder_optim, linear_optim]
        else:
            if len(linear_params) == 0:
                return torch.optim.AdamW(
                    self.parameters(),
                    lr=self.encoder_learning_rate,
                    weight_decay=self.weight_decay,
                )
            else:
                if len(model_params) == 0:
                    return [torch.optim.AdamW(
                        linear_params.values(),
                        lr=self.weights_learning_rate,
                    )]
                else:
                    encoder_optim = torch.optim.AdamW(
                        model_params.values(),
                        lr=self.encoder_learning_rate,
                        weight_decay=self.weight_decay,
                    )

                    linear_optim = torch.optim.AdamW(
                        linear_params.values(),
                        lr=self.weights_learning_rate,
                    )

                    return [encoder_optim, linear_optim]

    def forward(self, batch: DecomposedInstanceBatch) -> torch.Tensor:
        x_encoded = self.encoder(batch.query)["sentence_embedding"]  # R[Batch, Emb]
        x_pos_encoded = self.encoder(batch.pos_cand)["sentence_embedding"]  # R[Batch, Emb]
        x_neg_encoded = self.encoder(batch.neg_cands)["sentence_embedding"]  # R[Batch, Emb]
        return x_encoded, x_pos_encoded, x_neg_encoded

    def transfer_batch_to_device(
            self,
            batch: DecomposedInstanceBatch,
            device: torch.device, dataloader_idx: int
    ) -> DecomposedInstanceBatch:
        if batch.mode == Kind.QUERY:
            query = super().transfer_batch_to_device(batch.query[FieldType.DENSE], device=device, dataloader_idx=dataloader_idx)
            return InstanceBatch(batch.mode,
                                 query={FieldType.DENSE: query},
                                 pos_cand=None,
                                 neg_cands=None,
                                 instances=batch.instances)
        elif batch.mode == Kind.HYBRID:
            query, pos_cand, neg_cands = super().transfer_batch_to_device(
                (batch.query[FieldType.DENSE], batch.pos_cand[FieldType.DENSE], batch.neg_cands[FieldType.DENSE]),
                device=device, dataloader_idx=dataloader_idx
            )
            return DecomposedInstanceBatch(
                batch.mode,
                {FieldType.DENSE: query},
                {FieldType.DENSE: pos_cand},
                {FieldType.DENSE: neg_cands},
                instances=batch.instances
            )
        else:
            raise ValueError(f"Unknown batch type: {batch}")

    def encode_for_training(self, batch: InstanceBatch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_encoded = self.encoder(batch.query[FieldType.DENSE])["sentence_embedding"]
        dense_fields = [field.key for field in self.field_info.values() if field.field_type == FieldType.DENSE]
        if dense_fields:
            x_pos = [self.encoder(batch.pos_cand[FieldType.DENSE][field_key])["sentence_embedding"] for field_key in dense_fields]
            x_neg = [self.encoder(batch.neg_cands[FieldType.DENSE][field_key])["sentence_embedding"] for field_key in dense_fields]
            x_pos_encoded = torch.stack(x_pos, dim=1)
            x_neg_encoded = torch.stack(x_neg, dim=1)
        else:
            x_pos_encoded = torch.empty(x_encoded.size(0), 0, x_encoded.size(1), device=x_encoded.device)
            x_neg_encoded = torch.empty(x_encoded.size(0), 0, x_encoded.size(1), device=x_encoded.device)

        num_neg_samples = x_neg_encoded.size(0) // x_encoded.size(0)
        x_neg_encoded = x_neg_encoded.view(x_encoded.size(0), len(dense_fields), num_neg_samples, x_encoded.size(1))
        return x_encoded, x_pos_encoded, x_neg_encoded

    def compute_loss(self, batch: InstanceBatch, x_encoded: torch.Tensor, x_pos_encoded: torch.Tensor, x_neg_encoded: torch.Tensor) -> torch.Tensor:
        # Dims need to be: R[Batch, n_fields, Emb] for x_pos_encoded and R[Batch, n_fields, NegSample, Emb] for x_neg_encoded
        queries = [batch.instances[idx].query.text for idx in range(len(batch.instances))]
        any_field_key = list(self.field_info.keys())[0]
        query_ids = [int(batch.instances[idx].query._id) for idx in range(len(batch.instances))]
        pos_docs = [batch.instances[idx].pos_cand[any_field_key][0] for idx in range(len(batch.instances))]
        neg_docs = [batch.instances[idx].neg_cands[any_field_key][0][0] for idx in range(len(batch.instances))] # Assume only one negative
        queries = pickle.dumps(queries)
        pos_docs = pickle.dumps(pos_docs)
        neg_docs = pickle.dumps(neg_docs)
        query_ids = pickle.dumps(query_ids)
        loss = self.hybrid_contrastive_loss_fn(
            x_encoded,
            queries,
            x_pos_encoded,
            pos_docs,
            x_neg_encoded,
            neg_docs,
            query_ids,
            self.precomputed_sparse_scores
        )
        return loss

    def training_step(self, batch: InstanceBatch, batch_idx: int) -> torch.Tensor:
        opts = self.optimizers()
        if type(opts) is not list:
            opts = [opts]
        for opt in opts:
            opt.zero_grad()
        x_encoded, x_pos_encoded, x_neg_encoded = self.encode_for_training(batch)
        loss_c = self.compute_loss(batch, x_encoded, x_pos_encoded, x_neg_encoded)

        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            print(f"Training loss: {loss_c}")
        self.log("train_loss", loss_c.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=x_encoded.size(0))
        self.manual_backward(loss_c)
        for opt in opts:
            opt.step()
        return

    def on_eval_start(self) -> None:
        n = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        self.qres_output = open(f"{self.out_dir}/{rank}.qres", "w")

        corpus_segment = self.corpus[self.n_docs * rank // n: self.n_docs * (rank + 1) // n]
        dense_fields = [field.key for field in self.field_info.values() if field.field_type == FieldType.DENSE]

        all_field_name_docs = {
            field.key: format_documents(corpus_segment, field.name, field.dataset) for field in self.field_info.values()
        }
        if self.prefix:
            all_field_name_docs = {
                field.key: [
                    (_id, field.name + ": " + doc) for _id, doc in all_field_name_docs[field.key]
                ] for field in self.field_info.values()
            }
        for dense_field_key in dense_fields:
            for key, vec in candidate_encoding_stream(
                self.encoder,
                all_field_name_docs[dense_field_key],
                batch_size=self.dev_batch_size,
                multiprocess=False,
                show_progress=False,
            ):
                self.vectors_dict[dense_field_key][key] = vec
        torch.distributed.barrier()
        for dense_field_key in dense_fields:
            self.vectors_dict[dense_field_key].reopen()
            self.indices_dict[dense_field_key].vectors = self.vectors_dict[dense_field_key].file
        # Filter to only the fields we care about
        self.indices_dict = {k: self.indices_dict[k] for k in self.field_info.keys()}


    def on_validation_epoch_start(self) -> None:
        """
        Encodes the model for the entire corpus and builds the index.
        """
        if self.trec_val_freq > 0 and (self.trainer.current_epoch + 1) % self.trec_val_freq == 0:
            self.on_eval_start()

    def validation_step(self, batch: Union[InstanceBatch, DecomposedInstanceBatch], batch_idx: int, dataloader_idx: int = 0) -> Union[torch.tensor, None]:
        if (self.trec_val_freq > 0 and (self.trainer.current_epoch + 1) % self.trec_val_freq == 0
            and dataloader_idx == 1):
            self.trec_eval_step(batch, batch_idx, self.qres_output)

        # For consistency with stopping criteria, we always run the fast proxy validation.
        if dataloader_idx == 0:
            return self.proxy_validation_step(batch, dataloader_idx=dataloader_idx)


    def on_validation_epoch_end(self) -> None:
        if self.trec_val_freq > 0 and (self.trainer.current_epoch + 1) % self.trec_val_freq == 0:
            self.qres_output.close()
            torch.distributed.barrier()

            n = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
            processed_query_ids = set()  # dedup across GPUs (DDP problem)
            qres_files = [f"{self.out_dir}/{i}.qres" for i in range(n)]

            all_qres_file_name = f"{self.out_dir}/epoch-{self.global_step}-all-{rank}.qres"
            with open(all_qres_file_name, "w") as qres_output:
                for qres_file in qres_files:
                    with open(qres_file) as f:
                        query_ids = set()
                        qress = trec.QRes.from_text_io(f)
                        for qres in qress:
                            if qres.query_id not in processed_query_ids:
                                query_ids.add(qres.query_id)
                                print(qres, file=qres_output)
                    processed_query_ids.update(query_ids)

            metrics = trec.call_trec_eval_and_get_metrics(
                qrels=self.dev_qrels_path,
                qres=all_qres_file_name,
            )
            for metric, value in metrics.items():
                log_to_prog_bar = metric in {"map", "recip_rank", "recall_5", "recall_10", "recall_20", "recall_100", "ndcg", "ndcg_cut_10", "Rprec", "success_1"}
                if log_to_prog_bar and rank == 0:
                    print(f"{metric}: {value:.3f}")
                self.log(f"dev_{metric}", value, prog_bar=log_to_prog_bar, logger=True, sync_dist=True, add_dataloader_idx=False)

            new_score = metrics["ndcg_cut_10"]
            if self.global_step == 0 or new_score > self.best_score:
                self.best_score = new_score


    def on_test_epoch_start(self) -> None:
        self.on_eval_start()
        rank = torch.distributed.get_rank()
        self.additional_qres_output = open(f"{self.out_dir}/additional_{rank}.qres", "w")


    def test_step(self, batch: Union[InstanceBatch, DecomposedInstanceBatch], batch_idx: int, dataloader_idx: int = 0) -> Union[torch.tensor, None]:
        if dataloader_idx == 0:
            self.trec_eval_step(batch, batch_idx, self.qres_output)
        elif dataloader_idx == 1:
            self.trec_eval_step(batch, batch_idx, self.additional_qres_output)


    def merge_qres_and_score(self, qres_files, qrels_path, additional=""):
        rank = torch.distributed.get_rank()
        if rank != 0:
            return
        processed_query_ids = set()  # dedup across GPUs (DDP problem)
        all_qres_file_name = f"{self.out_dir}/final-{additional}all-{rank}.qres"
        with open(all_qres_file_name, "w") as qres_output:
            for qres_file in qres_files:
                with open(qres_file) as f:
                    query_ids = set()
                    qress = trec.QRes.from_text_io(f)
                    for qres in qress:
                        if qres.query_id not in processed_query_ids:
                            query_ids.add(qres.query_id)
                            print(qres, file=qres_output)
                processed_query_ids.update(query_ids)

        metrics = trec.call_trec_eval_and_get_metrics(
            qrels=qrels_path,
            qres=all_qres_file_name,
        )
        keys = ["success_1", "success_5", "recall_5", "recall_10", "recall_15", "recall_20", "ndcg", "ndcg_cut_10", "recip_rank", "map"]
        print("\t".join(keys))
        print("\t".join([f"{metrics[key]:.3f}" for key in keys]))
        result_string = json.dumps({
            "success_1": f"{metrics['success_1']:.3f}",
            "success_5": f"{metrics['success_5']:.3f}",
            "recall_5": f"{metrics['recall_5']:.3f}",
            "recall_10": f"{metrics['recall_10']:.3f}",
            "recall_15": f"{metrics['recall_15']:.3f}",
            "recall_20": f"{metrics['recall_20']:.3f}",
            "ndcg": f"{metrics['ndcg']:.3f}",
            "ndcg_cut_10": f"{metrics['ndcg_cut_10']:.3f}",
            "recip_rank": f"{metrics['recip_rank']:.3f}",
            "map": f"{metrics['map']:.3f}",
            "masked_fields": self.masked_fields_string,
            "additional": "test" if additional != "" else "val",
        })
        print(result_string)
        with open(f"{self.out_dir}/results_dicts-all-{rank}.jsonl", "a") as f:
            f.write(result_string + "\n")

        if additional == "":
            metrics = {f"best_{k}": v for k, v in metrics.items()}
            self.log_dict(metrics, sync_dist=True, logger=True, add_dataloader_idx=False, rank_zero_only=True)
        else:
            metrics = {f"additional_{k}": v for k, v in metrics.items()}
            self.log_dict(metrics, sync_dist=True, logger=True, add_dataloader_idx=False, rank_zero_only=True)


    def on_test_epoch_end(self) -> None:
        rank = torch.distributed.get_rank()
        self.qres_output.close()
        if os.path.getsize(f"{self.out_dir}/additional_{rank}.qres") == 0:
            has_additional = False
        else:
            self.additional_qres_output.close()
            has_additional = True
        torch.distributed.barrier()

        n = torch.distributed.get_world_size()
        qres_files = [f"{self.out_dir}/{i}.qres" for i in range(n)]
        self.merge_qres_and_score(qres_files, self.dev_qrels_path)
        if has_additional:
            additional_qres_files = [f"{self.out_dir}/additional_{i}.qres" for i in range(n)]
            self.merge_qres_and_score(additional_qres_files, self.additional_qrels_path, additional="additional-")


    def on_save_checkpoint(self, checkpoint):
        new_field_info = {field_name: field.serialize() for field_name, field in checkpoint["hyper_parameters"]["field_info"].items()}
        checkpoint["hyper_parameters"]["field_info"] = new_field_info
        checkpoint["hyper_parameters"]["indices_list"] = []
        checkpoint["hyper_parameters"]["vectors_list"] = []
        checkpoint["hyper_parameters"]["corpus"] = []
        checkpoint["hyper_parameters"]["precomputed_sparse_scores"] = []

    def on_load_checkpoint(self, checkpoint):
        new_field_info = {field_name: Field.deserialize(field) if isinstance(field, dict) else field
                          for field_name, field in checkpoint["hyper_parameters"]["field_info"].items()}
        checkpoint["hyper_parameters"]["field_info"] = new_field_info

    def proxy_validation_step(
        self,
        batch: Union[InstanceBatch, DecomposedInstanceBatch],
        dataloader_idx: int
    ):
        """
        Proxy step where we use the loss as the metric...
        """
        x_encoded, x_pos_encoded, x_neg_encoded = self.encode_for_training(batch)
        loss_c = self.compute_loss(batch, x_encoded, x_pos_encoded, x_neg_encoded)
        assert not loss_c.requires_grad
        self.log("valid_loss", loss_c.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, add_dataloader_idx=False)
        if torch.distributed.get_rank() == 0:
            print(f"Validation loss: {loss_c}")
            print(f"Cache information: BM25s tokenize: {BM25sSparseIndex.tokenize_single.cache_info()}")
            index_memo = [f"{field_name}: {index.get_scores.cache_info()}"
                            for field_name, index in self.indices_dict.items()
                            if isinstance(index, BM25sSparseIndex)]
            index_memo_str = index_memo[0] if index_memo else "None"
            print(f"Cache information: BM25sSparseIndex index memo: {index_memo_str}")
        return loss_c

    def trec_eval_step(self, batch: QueryDataset, batch_idx: int, qres_output) -> None:
        data = batch.instances
        all_hits = []
        for _, index in self.indices_dict.items():
            hits = index.retrieve_batch([x.text for x in data], top_k=100)
            all_hits.append(hits)

        hits_ids = np.array([[[h[0] for h in hit] for hit in field] for field in all_hits])
        for i, q in zip(range(len(hits_ids[0])), data):
            ids_set = [set(hits) for hits in hits_ids[:, i, :].tolist()]
            all_ids_set = list(reduce(lambda x, y: x | y, ids_set))
            new_hits = []
            for index in self.indices_dict.values():
                hits = index.score_batch([q.text], all_ids_set) # [1 x num ids]
                new_hits.append(hits)

            all_tens = torch.stack(new_hits, dim=0).squeeze(1).cuda() # [num_indices x num_ids]
            all_tens = all_tens * self.mask.cuda()

            q_toks = {
                'input_ids': batch.query[FieldType.DENSE]['input_ids'][i][:self.encoder.get_max_seq_length()].unsqueeze(0),
                'attention_mask': batch.query[FieldType.DENSE]['attention_mask'][i][:self.encoder.get_max_seq_length()].unsqueeze(0),
            }

            x_encoded = self.encoder(q_toks)["sentence_embedding"]
            scores = self.hybrid_contrastive_loss_fn.mixture_of_fields_layer(all_tens.t(), x_encoded)

            values, indices = torch.topk(scores, k=100, dim=1)

            # Get the chosen indices and calculate best docs
            chosen_indices = indices.cpu().flatten().tolist()
            documents = [all_ids_set[idx] for idx in chosen_indices]
            values = values.squeeze()
            for value, index in zip(values, documents):
                qres = trec.QRes(query_id=q._id, doc_id=index, sim=value.item())
                print(qres, file=qres_output)

    def mask_field(self, field_idx_list):
        field_names = list(self.field_info.keys())
        masked_fields = [field_names[field_idx] for field_idx in field_idx_list]
        if torch.distributed.get_rank() == 0:
            print(f"Masking fields: {masked_fields}")
        self.masked_fields_string = ",".join(masked_fields)
        mask = torch.ones([len(self.field_info), 1], device=self.device)
        mask[field_idx_list] = 0
        self.mask = mask