from concurrent.futures import ThreadPoolExecutor
from itertools import chain
from typing import Dict, Optional, List, Tuple

import pickle
import torch
try:
    import torch.distributed.nn.functional as dist_F
except:
    pass

class BaseContrastiveLoss(torch.nn.Module):
    def __init__(
        self,
        temperature: float = 0.01,
        in_batch_negative: bool = True,
        reverse: bool = True,
        all_gather_multi_gpu: bool = True,
        multi_fields: bool = False,
    ):
        super().__init__()
        self.temperature = temperature
        self.in_batch_negative = in_batch_negative
        self.reverse = reverse
        self.all_gather_multi_gpu = all_gather_multi_gpu
        self.multi_fields = multi_fields

    # This is a base class, these functions need to be implemented by the subclasses:
    def forward(self,
                q: torch.Tensor,
                d_pos: torch.Tensor,
                d_neg: Optional[torch.Tensor],
    ):
        raise NotImplementedError

    def compute_query_doc_scores(self, q, d_pos, d_neg):
        raise NotImplementedError

    def compute_doc_query_scores(self, d_pos, q):
        raise NotImplementedError

    def gather_all_embeddings(self, q, d_pos, d_neg, use_multi_gpu):
        # Gather from all GPUs, effectively increasing the batch size by the number of GPUs
        # for in-batch negative sampling
        if use_multi_gpu:
            all_q = dist_F.all_gather(q)
            all_d_pos = dist_F.all_gather(d_pos)
            all_d_neg = dist_F.all_gather(d_neg)
            q = torch.cat(all_q, dim=0)
            d_pos = torch.cat(all_d_pos, dim=0)
            d_neg = torch.cat(all_d_neg, dim=0)
        return q, d_pos, d_neg

    def distributed_reduce_mean_nll(self, nll, use_multi_gpu):
        if use_multi_gpu:
            nll = dist_F.all_reduce(nll) / torch.distributed.get_world_size()
        return nll

    def sliced_nll(self, scores, batch_size, gpu_id):
        log_probs = torch.log_softmax(scores, dim=1) # R[Batch, Devices*Batch + ...]
        # Be careful to get the correct diagonal
        sliced_scores = log_probs[:, (batch_size * gpu_id): (batch_size * (gpu_id + 1))]  # R[Batch, Batch]
        log_probs = torch.diag(sliced_scores) # R[Batch]
        nll = -torch.mean(log_probs)
        return nll

    def in_batch_negative_loss(self, q, d_pos, d_neg, use_multi_gpu):
        """
        q: [Batch, Emb]
        d_pos: [Batch, Emb] or [Batch, Field, Emb]
        d_neg: [Batch, NegSample, Emb] or [Batch, Field, NegSample, Emb]
        """
        all_q, all_d_pos, all_d_neg = self.gather_all_embeddings(q, d_pos, d_neg, use_multi_gpu)
        scores_pos, scores_neg = self.compute_query_doc_scores(q, all_d_pos, all_d_neg)
        all_scores = torch.cat([scores_pos, scores_neg], dim=1)  # R[Batch, Devices*Batch + ...]

        per_device_batch_size = q.size(0)
        gpu_id = torch.distributed.get_rank() if use_multi_gpu else 0
        nll = self.sliced_nll(all_scores, per_device_batch_size, gpu_id)

        if self.reverse:
            rev_scores = self.compute_doc_query_scores(d_pos, all_q)
            rev_nll = self.sliced_nll(rev_scores, per_device_batch_size, gpu_id)
            nll += rev_nll
        return nll

    def simple_loss(self, q, d_pos, d_neg):
        """
        q: [Batch, Emb]
        d_pos: [Batch, Emb] or [Batch, Field, Emb]
        d_neg: [Batch, NegSample, Emb] or [Batch, NegSample, Field, Emb]
        """
        if self.multi_fields:
            # R[Batch, Field, Emb] * R[Batch, Emb, 1] -> R[Batch, Field, 1] -> R[Batch, 1, Field]
            scores_pos_per_field = torch.matmul(d_pos, q.unsqueeze(2)).squeeze(2).unsqueeze(1) / self.temperature
            # R[Batch, 1, 1, Emb] * R[Batch, NegSample, Field, Emb] -> R[Batch, NegSample, Field]
            scores_neg_per_field = torch.sum(q.view(q.size(0), 1, 1, q.size(1)) * d_neg.permute(0, 2, 1, 3), dim=-1) / self.temperature
            scores_pos = self.mixture_of_fields_layer(scores_pos_per_field, q) # R[Batch, 1]
            scores_neg = self.mixture_of_fields_layer(scores_neg_per_field, q) # R[Batch, NegSample]
        else:
            # R[Batch, 1, Emb] * [Batch, Emb, 1] -> R[Batch, 1, 1]
            scores_pos = torch.matmul(q.unsqueeze(1), d_pos.unsqueeze(2)).squeeze(1) / self.temperature
            # R[Batch, 1, Emb] * [Batch, Emb, NegSample] -> R[Batch, 1, NegSample]
            scores_neg = torch.matmul(q.unsqueeze(1), d_neg.transpose(1, 2)).squeeze(1) / self.temperature

        all_scores = torch.cat([scores_pos, scores_neg], dim=1)  # R[Batch, 1+TotalNegSamples]
        log_probs = torch.log_softmax(all_scores, dim=1)
        pos_log_probs = log_probs[:, 0]
        nll = -torch.mean(pos_log_probs)
        return nll


class ContrastiveLoss(BaseContrastiveLoss):
    def __init__(
        self,
        temperature: float = 0.01,
        in_batch_negative: bool = True,
        reverse: bool = True,
        all_gather_multi_gpu: bool = True,
    ):
        super().__init__(temperature, in_batch_negative, reverse, all_gather_multi_gpu)

    def forward(
        self,
            q: torch.Tensor,  # R[Batch, Emb]
            d_pos: torch.Tensor,  # R[Batch, Emb]
            d_neg: Optional[torch.Tensor],  # R[Batch, NegSample, Emb]
    ) -> torch.Tensor:  # R[]
        use_multi_gpu = self.all_gather_multi_gpu and torch.distributed.is_initialized()
        if self.in_batch_negative:
            nll = self.in_batch_negative_loss(q, d_pos, d_neg, use_multi_gpu)
        else:
            nll = self.simple_loss(q, d_pos, d_neg)
        return self.distributed_reduce_mean_nll(nll, use_multi_gpu)

    def compute_query_doc_scores(self, q, d_pos, d_neg):
        # R[Batch, Emb] * [Emb, Devices*Batch] -> R[Batch, Devices*Batch]
        scores_pos = torch.matmul(q, d_pos.t())/ self.temperature
        emb_dim = d_neg.size(-1)
        # R[Batch, Emb] * [Emb, Devices*Batch*NegSample] -> R[Batch, Devices*Batch*NegSample]
        scores_neg = torch.matmul(q, d_neg.view(-1, emb_dim).t()) / self.temperature
        return scores_pos, scores_neg

    def compute_doc_query_scores(self, d_pos, q):
        # R[Batch, Emb] * [Emb, Devices*Batch] -> R[Batch, Devices*Batch]
        return torch.matmul(d_pos, q.t())/ self.temperature


class DecomposedContrastiveLoss(BaseContrastiveLoss):
    def __init__(
        self,
        temperature: float = 0.01,
        in_batch_negative: bool = True,
        reverse: bool = True,
        all_gather_multi_gpu: bool = True,
        mixture_of_fields_layer: torch.nn.Module = None,
    ):
        super().__init__(temperature, in_batch_negative, reverse, all_gather_multi_gpu, multi_fields=True)
        self.mixture_of_fields_layer = mixture_of_fields_layer

    def forward(
        self,
        q: torch.Tensor,  # R[Batch, Emb]
        d_pos: torch.Tensor,  # R[Batch, Field, Emb]
        d_neg: Optional[torch.Tensor],  # R[Batch, Field, NegSample, Emb]
    ) -> torch.Tensor:
        use_multi_gpu = self.all_gather_multi_gpu and torch.distributed.is_initialized()

        if self.in_batch_negative:
            nll = self.in_batch_negative_loss(q, d_pos, d_neg, use_multi_gpu)
        else:
            nll = self.simple_loss(q, d_pos, d_neg)

        return self.distributed_reduce_mean_nll(nll, use_multi_gpu)

    def compute_query_doc_field_components(
            self,
            q, # R[Batch, Emb]
            d_pos, # R[Batch, Field, Emb]
            d_neg, # R[Batch, Field, NegSample, Emb]
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            reshaped_q = q.unsqueeze(1).unsqueeze(3)  # R[Batch, 1, Emb, 1]
            reshaped_d_pos = d_pos.unsqueeze(0) # R[1, Devices*Batch, Field, Emb]
            scores_pos_components = (torch.matmul(reshaped_d_pos, reshaped_q) / self.temperature).squeeze(3) # R[1, Devices*Batch, Field, 1] -> R[1, Devices*Batch, Field, 1]
            (total_batch_size, num_fields, neg_samples_per_query, emb_dim) = tuple(d_neg.shape)
            reshaped_all_d_neg = d_neg.permute(0, 2, 1, 3).view(1, total_batch_size * neg_samples_per_query, num_fields, emb_dim) # R[1, Devices * batch * NegSample, Field, Emb]
            scores_neg_components = (torch.matmul(reshaped_all_d_neg, reshaped_q) / self.temperature).squeeze(3) # R[Batch, Devices * Batch * NegSample, Field]
            return scores_pos_components, scores_neg_components

    def compute_query_doc_scores(
            self,
            q, # R[Batch, Emb]
            d_pos, # R[Devices * Batch, Field, Emb]
            d_neg, # R[Devices * Batch, Field, NegSample, Emb]
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            scores_pos_components, scores_neg_components = self.compute_query_doc_field_components(q, d_pos, d_neg)
            scores_pos = self.mixture_of_fields_layer(scores_pos_components, q) # R[Batch, Devices*Batch]
            scores_neg = self.mixture_of_fields_layer(scores_neg_components, q) # R[Batch, Devices*Batch*NegSamples]
            return scores_pos, scores_neg

    def compute_doc_query_scores(self, d_pos, q):
        rev_scores_components = torch.matmul(d_pos, q.t().unsqueeze(0)) / self.temperature # R[Batch, Field, Devices*Batch]
        # R[Devices * Batch, Batch, Field] , R[Devices * Batch, Emb] -> R[Devices * Batch, Batch] -> R[Batch, Devices*Batch]
        return self.mixture_of_fields_layer(rev_scores_components.permute(2, 0, 1), q).t()

class HybridContrastiveLoss(DecomposedContrastiveLoss):
    def __init__(
        self,
        temperature: float = 0.01,
        in_batch_negative: bool = True,
        reverse: bool = True,
        all_gather_multi_gpu: bool = True,
        mixture_of_fields_layer: torch.nn.Module = None,
        sparse_indices_dict: Dict = None,
        num_fields: int = 0,
        use_batchnorm = False,
    ):
        super().__init__(temperature, in_batch_negative, reverse, all_gather_multi_gpu, mixture_of_fields_layer)
        self.sparse_indices_dict = sparse_indices_dict
        if use_batchnorm:
            self.bn = torch.nn.BatchNorm1d(num_fields, track_running_stats=True) # need to try false
        else:
            self.bn = torch.nn.Identity()

    def forward(
        self,
        q: torch.Tensor,  # R[Batch, Emb]
        queries: List[str], # List[Batch]
        d_pos: torch.Tensor,  # R[Batch, Field, Emb]
        pos_docs: List[str], # List[Batch]
        d_neg: Optional[torch.Tensor],  # R[Batch, Field, NegSample, Emb]
        neg_docs: Optional[List[str]], # List[Batch]
        query_ids: List[int],
        sparse_scores: Optional[Dict] = None,
    ) -> torch.Tensor:
        use_multi_gpu = self.all_gather_multi_gpu and torch.distributed.is_initialized()
        if self.in_batch_negative:
            nll = self.in_batch_negative_loss(q, queries, d_pos, pos_docs, d_neg, neg_docs, use_multi_gpu, query_ids, sparse_scores)
        else:
            nll = self.simple_loss(q, d_pos, d_neg)
        return self.distributed_reduce_mean_nll(nll, use_multi_gpu)

    def gather_all_embeddings(
        self,
        q,
        queries,
        query_ids,
        d_pos,
        pos_docs,
        d_neg,
        neg_docs,
        use_multi_gpu,
    ):
        if use_multi_gpu:
            all_q = dist_F.all_gather(q)
            all_d_pos = dist_F.all_gather(d_pos)
            all_d_neg = dist_F.all_gather(d_neg)
            q = torch.cat(all_q, dim=0)
            d_pos = torch.cat(all_d_pos, dim=0)
            d_neg = torch.cat(all_d_neg, dim=0)
            all_queries = [None for _ in range(torch.distributed.get_world_size())]
            all_pos_docs = [None for _ in range(torch.distributed.get_world_size())]
            all_neg_docs = [None for _ in range(torch.distributed.get_world_size())]
            all_query_ids = [None for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather_object(all_queries, pickle.loads(queries))
            torch.distributed.all_gather_object(all_pos_docs, pickle.loads(pos_docs))
            torch.distributed.all_gather_object(all_neg_docs, pickle.loads(neg_docs))
            torch.distributed.all_gather_object(all_query_ids, pickle.loads(query_ids))
            flattened_queries = list(chain(*all_queries))
            flattened_pos_docs = list(chain(*all_pos_docs))
            flattened_neg_docs = list(chain(*all_neg_docs))
            flattened_query_ids = list(chain(*all_query_ids))
        return q, flattened_queries, d_pos, flattened_pos_docs, d_neg, flattened_neg_docs, flattened_query_ids

    def in_batch_negative_loss(self, q, queries, d_pos, pos_docs, d_neg, neg_docs, use_multi_gpu, query_ids, sparse_scores):
        """
        q: [Batch, Emb]
        d_pos: [Batch, Emb] or [Batch, Field, Emb]
        d_neg: [Batch, NegSample, Emb] or [Batch, Field, NegSample, Emb]
        """
        all_q, all_queries_text, all_d_pos, all_pos_ids, all_d_neg, all_neg_ids, all_query_ids = self.gather_all_embeddings(
            q, queries, query_ids, d_pos, pos_docs, d_neg, neg_docs, use_multi_gpu
        )

        queries = pickle.loads(queries) # For gpu-specific use
        query_ids = pickle.loads(query_ids)
        pos_docs = pickle.loads(pos_docs) # For gpu-specific use
        scores_pos, scores_neg = self.compute_query_doc_scores(
            q, queries, all_d_pos, all_pos_ids, all_d_neg, all_neg_ids, query_ids, sparse_scores
        )
        all_scores = torch.cat([scores_pos, scores_neg], dim=1)  # R[Batch, Devices*Batch + ...]

        per_device_batch_size = q.size(0)
        gpu_id = torch.distributed.get_rank() if use_multi_gpu else 0
        nll = self.sliced_nll(all_scores, per_device_batch_size, gpu_id)

        if self.reverse:
            rev_scores = self.compute_doc_query_scores(d_pos, pos_docs, all_q, all_queries_text, all_query_ids, sparse_scores)
            rev_nll = self.sliced_nll(rev_scores, per_device_batch_size, gpu_id)
            nll += rev_nll
        return nll

    def compute_sparse_query_doc_scores(
        self,
        queries, #[QueryBatch]
        doc_ids, # [DocBatch]
        query_ids: List[int], # [QueryBatch]
        sparse_scores: Optional[Dict],
    ): #-> R[QueryBatch, DocBatch * Devices, num_fields]:
        """
        If we have scores saved in sparse_scores, then use that instead of calculating bm25 during the forward pass

        However at dev/test-time, those scores do not exist and so we will need to expensively compute them.
        """
        if len(self.sparse_indices_dict) > 0:
            if any([all([qid in sparse_score_by_field for qid in query_ids])
                    for sparse_score_by_field in sparse_scores.values()]):
                per_field_scores = [si.score_batch_with_cache(query_ids, doc_ids, sparse_scores[field_name])
                                    for field_name, si in self.sparse_indices_dict.items()]
            else:
                with ThreadPoolExecutor(max_workers=len(self.sparse_indices_dict)) as executor:
                    per_field_scores = list(executor.map(lambda si: si.score_batch(queries, doc_ids), self.sparse_indices_dict.values()))
            return torch.stack(per_field_scores, dim=-1).cuda()
        else:
            return torch.empty(len(queries), len(doc_ids), 0).cuda()

    def compute_query_doc_scores(
        self,
        q, # R[QueryBatch, Emb]
        queries, # R[QueryBatch]
        d_pos, # R[DocBatch, Field, Emb]
        pos_text, # R[Field, DocBatch]
        d_neg, # R [DocBatch, Field, NegSample, Emb]
        neg_text, # R[Field, DocBatch]
        query_ids, # List[int]
        sparse_scores, # Dict
    ):
        dense_scores_pos, dense_scores_neg = self.compute_query_doc_field_components(q, d_pos, d_neg)
        sparse_scores_pos = self.compute_sparse_query_doc_scores(queries, pos_text, query_ids, sparse_scores)
        sparse_scores_neg = self.compute_sparse_query_doc_scores(queries, neg_text, query_ids, sparse_scores)

        all_scores_pos = torch.cat([dense_scores_pos, sparse_scores_pos], dim=-1)
        all_scores_neg = torch.cat([dense_scores_neg, sparse_scores_neg], dim=-1)

        all_scores = torch.cat([all_scores_pos, all_scores_neg], dim=1) # R[QueryBatch, DocBatch * Devices, num_fields]
        all_scores_normed = self.bn(all_scores.permute(0, 2, 1)).permute(0, 2, 1)
        all_scores_combined = self.mixture_of_fields_layer(all_scores_normed, q) # R[QueryBatch, DocBatch * Devices]
        scores_pos = all_scores_combined[:, :dense_scores_pos.size(1)]
        scores_neg = all_scores_combined[:, dense_scores_pos.size(1):]
        return scores_pos, scores_neg

    def compute_doc_query_scores(self, d_pos, pos_docs, q, queries, query_ids, sparse_scores):
        # There's some unnecessary transposes going on here... not going to touch that for now
        dense_rev_scores = torch.matmul(d_pos, q.t().unsqueeze(0)) / self.temperature # R[DocBatch, Field, Devices*QueryBatch]
        dense_rev_scores = dense_rev_scores.permute(2, 0, 1) #[Devices*QueryBatch, DocBatch, Field]
        sparse_rev_scores = self.compute_sparse_query_doc_scores(queries, pos_docs, query_ids, sparse_scores) # R[QueryBatch * Devices, Doc_Batch, num_fields]

        all_scores = torch.concat([dense_rev_scores, sparse_rev_scores], dim=2) # R[QueryBatch * Devices, DocBatch, num_dense+num_sparse fields]
        all_scores = self.bn(all_scores.permute(0, 2, 1)).permute(0, 2, 1) # R[QueryBatch * Devices, DocBatch, num_dense+num_sparse fields]
        return self.mixture_of_fields_layer(all_scores, q).t() # R[Batch, Devices*Batch]
