import logging
import time
from typing import List, Optional

import numpy as np
import torch
from sgl_kernel.speculative import reconstruct_indices_from_tree_mask

from sglang.srt.environ import envs
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.sampler import get_token_ids_logprobs, get_top_logprobs
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.cpp_ngram.ngram_cache import NgramCache
from sglang.srt.speculative.ngram_info import NgramVerifyInput
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from triepilot.sglang_integration.budget import (
    TriePilotStrategyBank,
    bucketize_triepilot_draft_budgets,
    observe_triepilot_accept_lengths,
    observe_triepilot_strategy_feedback,
    resolve_triepilot_draft_budgets,
)
from triepilot.sglang_integration.features import compute_ngram_tree_features
from triepilot.sglang_integration.recorder import TriePilotNgramRecorder

logger = logging.getLogger(__name__)


USE_FULL_MASK = True


class NGRAMWorker:
    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        self.target_worker = target_worker
        self.model_runner = target_worker.model_runner
        self.tp_rank = tp_rank
        self.page_size = server_args.page_size
        self.draft_token_num: int = server_args.speculative_num_draft_tokens
        self.branch_length: int = server_args.speculative_ngram_branch_length
        self.max_match_window_size: int = (
            server_args.speculative_ngram_max_match_window_size
        )
        self.triepilot_step_id = 0
        self.triepilot_structural_features = None
        self.triepilot_allocation_metadata = None
        self.triepilot_controller_time_ns = 0
        self.triepilot_allocation_policy = server_args.triepilot_allocation_policy
        self.triepilot_batch_budget = server_args.triepilot_batch_budget
        self.triepilot_random_seed = server_args.triepilot_random_seed
        self.triepilot_accept_ema_alpha = server_args.triepilot_accept_ema_alpha
        self.triepilot_shape_buckets = server_args.triepilot_shape_buckets
        self.triepilot_shape_bucket_mode = server_args.triepilot_shape_bucket_mode
        self.triepilot_verify_draft_token_num = self.draft_token_num
        self.triepilot_requested_draft_budgets = None
        self.triepilot_bucketed_draft_budgets = None
        self.triepilot_active_draft_lengths = None
        self.triepilot_bucket_ids = None
        self.triepilot_bucket_padding_nodes = None
        self.triepilot_shape_padding_tokens = None
        self.triepilot_strategy_bank = TriePilotStrategyBank(
            server_args.speculative_num_draft_tokens
        )
        self.triepilot_recorder = TriePilotNgramRecorder(
            server_args.triepilot_trace_path,
            server_args.triepilot_run_id,
        )

        self.max_batch_size = target_worker.max_running_requests
        self.device = f"cuda:{gpu_id}" if gpu_id >= 0 else "cuda"

        self._init_preallocated_tensors()

        self.ngram_cache = NgramCache(
            min_match_window_size=server_args.speculative_ngram_min_match_window_size,
            max_match_window_size=server_args.speculative_ngram_max_match_window_size,
            min_bfs_breadth=server_args.speculative_ngram_min_bfs_breadth,
            max_bfs_breadth=server_args.speculative_ngram_max_bfs_breadth,
            capacity=server_args.speculative_ngram_capacity,
            branch_length=server_args.speculative_ngram_branch_length,
            draft_token_num=server_args.speculative_num_draft_tokens,
        )

    def clear_cache_pool(self):
        self.ngram_cache.reset()

    def _efficient_concat_last_n(self, seq1: List[int], seq2: List[int], n: int):
        seq2_len = len(seq2)
        if seq2_len >= n:
            return seq2[-n:]

        need_from_seq1 = n - seq2_len
        return seq1[-need_from_seq1:] + seq2

    def _init_preallocated_tensors(self):
        max_total_drafts = self.max_batch_size * self.draft_token_num
        max_total_mask_size = (
            self.max_batch_size * self.draft_token_num * self.draft_token_num
        )

        self.draft_tokens = torch.empty(
            (max_total_drafts,), dtype=torch.int64, device=self.device
        )
        self.retrieve_indexes = torch.empty(
            (self.max_batch_size, self.draft_token_num),
            dtype=torch.int64,
            device=self.device,
        )
        self.retrive_next_token = torch.empty(
            (self.max_batch_size, self.draft_token_num),
            dtype=torch.int64,
            device=self.device,
        )
        self.retrive_next_sibling = torch.empty(
            (self.max_batch_size, self.draft_token_num),
            dtype=torch.int64,
            device=self.device,
        )
        self.positions = torch.empty(
            (max_total_drafts,), dtype=torch.int64, device=self.device
        )
        self.tree_mask = torch.empty(
            (max_total_mask_size,), dtype=torch.bool, device=self.device
        )

        self.draft_tokens_batch = []
        self.tree_mask_batch = []
        self.retrieve_indexes_batch = []
        self.retrive_next_token_batch = []
        self.retrive_next_sibling_batch = []
        self.positions_batch = []

        for bs in range(0, self.max_batch_size + 1):
            self.retrieve_indexes_batch.append(self.retrieve_indexes[:bs, :])
            self.retrive_next_token_batch.append(self.retrive_next_token[:bs, :])
            self.retrive_next_sibling_batch.append(self.retrive_next_sibling[:bs, :])
            self.positions_batch.append(self.positions[: bs * self.draft_token_num])
            self.draft_tokens_batch.append(
                self.draft_tokens[: bs * self.draft_token_num]
            )
            self.tree_mask_batch.append(
                self.tree_mask[: bs * self.draft_token_num * self.draft_token_num]
            )

    def _prepare_draft_tokens(self, batch: ScheduleBatch) -> tuple[np.ndarray, np.ndarray]:
        bs = batch.batch_size()

        self.ngram_cache.synchronize()
        batch_tokens = []
        for req in batch.reqs:
            check_token = self._efficient_concat_last_n(
                req.origin_input_ids, req.output_ids, self.max_match_window_size
            )
            batch_tokens.append(check_token)
        req_drafts, mask = self.ngram_cache.batch_get(batch_tokens)
        total_draft_token_num = len(req_drafts)

        # Check if speculative decoding is needed; here we always enforce it
        assert (
            total_draft_token_num == bs * self.draft_token_num
        ), f"{total_draft_token_num=}, {bs=}, {self.draft_token_num=}"
        return req_drafts, mask

    def _build_variable_draft_inputs(
        self,
        batch: ScheduleBatch,
        req_drafts: np.ndarray,
        mask: np.ndarray,
        active_draft_lengths: list[int],
    ) -> dict[str, np.ndarray]:
        bs = batch.batch_size()
        full_drafts = req_drafts.reshape(bs, self.draft_token_num)
        full_masks = mask.reshape(bs, self.draft_token_num, self.draft_token_num)

        flat_drafts = []
        full_attention_masks = []
        positions = []
        retrive_index = []
        retrive_next_token = []
        retrive_next_sibling = []
        offset = 0

        for i, req in enumerate(batch.reqs):
            draft_len = int(active_draft_lengths[i])
            compact_mask = full_masks[i, :draft_len, :draft_len].astype(bool)
            flat_drafts.extend(full_drafts[i, :draft_len].tolist())

            seq_len = len(req.origin_input_ids) + len(req.output_ids)
            prefix_mask = np.ones((draft_len, max(seq_len - 1, 0)), dtype=bool)
            full_attention_masks.append(
                np.concatenate((prefix_mask, compact_mask), axis=1).reshape(-1)
            )

            for tid in range(draft_len):
                depth = 0
                parent_idx = -1
                for j in range(tid - 1, -1, -1):
                    if compact_mask[tid, j]:
                        depth += 1
                        if parent_idx == -1:
                            parent_idx = j

                positions.append(depth + seq_len)
                retrive_index.append(offset + tid)

                next_token_idx = -1
                for j in range(tid + 1, draft_len):
                    if compact_mask[j, tid]:
                        next_token_idx = j
                        break
                retrive_next_token.append(next_token_idx)

                next_sibling_idx = -1
                if parent_idx != -1:
                    for j in range(tid + 1, draft_len):
                        if compact_mask[j, parent_idx]:
                            is_sibling = True
                            for k in range(parent_idx + 1, j):
                                if compact_mask[j, k]:
                                    is_sibling = False
                                    break
                            if is_sibling:
                                next_sibling_idx = j
                                break
                retrive_next_sibling.append(next_sibling_idx)

            offset += draft_len

        return {
            "draft_tokens": np.asarray(flat_drafts, dtype=np.int64),
            "tree_mask": np.concatenate(full_attention_masks).astype(bool),
            "positions": np.asarray(positions, dtype=np.int64),
            "retrive_index": np.asarray(retrive_index, dtype=np.int64),
            "retrive_next_token": np.asarray(retrive_next_token, dtype=np.int64),
            "retrive_next_sibling": np.asarray(retrive_next_sibling, dtype=np.int64),
        }

    def _build_fixed_shape_draft_inputs(
        self,
        batch: ScheduleBatch,
        req_drafts: np.ndarray,
        mask: np.ndarray,
        active_draft_lengths: list[int],
        shape_draft_token_num: int,
    ) -> dict[str, torch.Tensor | int]:
        bs = batch.batch_size()
        shape_draft_token_num = max(1, int(shape_draft_token_num))
        full_drafts = req_drafts.reshape(bs, self.draft_token_num)
        full_masks = mask.reshape(bs, self.draft_token_num, self.draft_token_num)

        fixed_drafts = full_drafts[:, :shape_draft_token_num].copy()
        fixed_masks = full_masks[
            :, :shape_draft_token_num, :shape_draft_token_num
        ].copy()

        for index, active_length in enumerate(active_draft_lengths):
            active_length = min(max(int(active_length), 1), shape_draft_token_num)
            if active_length < shape_draft_token_num:
                fixed_masks[index, active_length:, :] = False
                fixed_masks[index, :, active_length:] = False
                for padding_idx in range(active_length, shape_draft_token_num):
                    fixed_masks[index, padding_idx, padding_idx] = True

        draft_tokens = torch.from_numpy(
            fixed_drafts.reshape(-1).astype(np.int64)
        ).to(device=self.device, non_blocking=True)
        compact_tree_mask = torch.from_numpy(
            fixed_masks.reshape(-1).astype(bool)
        ).to(device=self.device, non_blocking=True)
        positions = torch.empty(
            (bs * shape_draft_token_num,), dtype=torch.int64, device=self.device
        )
        retrive_index = torch.empty(
            (bs, shape_draft_token_num), dtype=torch.int64, device=self.device
        )
        retrive_next_token = torch.empty(
            (bs, shape_draft_token_num), dtype=torch.int64, device=self.device
        )
        retrive_next_sibling = torch.empty(
            (bs, shape_draft_token_num), dtype=torch.int64, device=self.device
        )

        reconstruct_indices_from_tree_mask(
            compact_tree_mask,
            batch.seq_lens,
            positions,
            retrive_index,
            retrive_next_token,
            retrive_next_sibling,
            bs,
            shape_draft_token_num,
        )

        if USE_FULL_MASK:
            full_attention_masks = []
            for index, req in enumerate(batch.reqs):
                seq_len = len(req.origin_input_ids) + len(req.output_ids)
                prefix_mask = np.ones(
                    (shape_draft_token_num, max(seq_len - 1, 0)),
                    dtype=bool,
                )
                full_attention_masks.append(
                    np.concatenate(
                        (prefix_mask, fixed_masks[index].astype(bool)),
                        axis=1,
                    ).reshape(-1)
                )
            tree_mask = torch.from_numpy(
                np.concatenate(full_attention_masks).astype(bool)
            ).to(device=self.device, non_blocking=True)
        else:
            tree_mask = compact_tree_mask

        return {
            "draft_tokens": draft_tokens,
            "tree_mask": tree_mask,
            "positions": positions,
            "retrive_index": retrive_index,
            "retrive_next_token": retrive_next_token,
            "retrive_next_sibling": retrive_next_sibling,
            "shape_draft_token_num": shape_draft_token_num,
        }

    def _prepare_for_speculative_decoding(self, batch: ScheduleBatch):
        if batch.forward_mode.is_extend():
            return

        bs = batch.batch_size()
        retrive_index = self.retrieve_indexes_batch[bs]
        retrive_next_token = self.retrive_next_token_batch[bs]
        retrive_next_sibling = self.retrive_next_sibling_batch[bs]
        positions = self.positions_batch[bs]
        tree_mask = self.tree_mask_batch[bs]
        draft_tokens = self.draft_tokens_batch[bs]

        req_drafts, mask = self._prepare_draft_tokens(batch)
        full_structural_features = compute_ngram_tree_features(
            req_drafts=req_drafts,
            mask=mask,
            draft_token_num=self.draft_token_num,
            active_draft_lengths=[self.draft_token_num] * bs,
        )
        controller_start_ns = time.perf_counter_ns()
        (
            requested_draft_budgets,
            active_draft_lengths,
            self.triepilot_allocation_metadata,
        ) = resolve_triepilot_draft_budgets(
            batch.reqs,
            self.draft_token_num,
            allocation_policy=self.triepilot_allocation_policy,
            batch_budget=self.triepilot_batch_budget,
            structural_features=full_structural_features,
            step_id=self.triepilot_step_id,
            random_seed=self.triepilot_random_seed,
            strategy_bank=self.triepilot_strategy_bank,
            return_metadata=True,
        )
        self.triepilot_controller_time_ns = (
            time.perf_counter_ns() - controller_start_ns
        )

        bucket_info = None
        bucketed_draft_budgets = list(requested_draft_budgets)
        bucket_ids = ["0/1" if budget <= 0 else str(budget) for budget in requested_draft_budgets]
        bucket_padding_nodes = [0] * len(requested_draft_budgets)
        shape_padding_tokens = [0] * len(requested_draft_budgets)
        verify_draft_token_num = self.draft_token_num
        if self.triepilot_shape_bucket_mode != "off":
            bucket_info = bucketize_triepilot_draft_budgets(
                requested_draft_budgets,
                default_budget=self.draft_token_num,
                bucket_spec=self.triepilot_shape_buckets or "default",
            )
            requested_draft_budgets = bucket_info["requested_budgets"]
            bucketed_draft_budgets = bucket_info["bucketed_budgets"]
            active_draft_lengths = bucket_info["active_draft_lengths"]
            bucket_ids = bucket_info["bucket_ids"]
            bucket_padding_nodes = bucket_info["bucket_padding_nodes"]

        if self.triepilot_shape_bucket_mode == "batch_max":
            verify_draft_token_num = max(active_draft_lengths) if active_draft_lengths else 1
            shape_padding_tokens = [
                max(verify_draft_token_num - int(length), 0)
                for length in active_draft_lengths
            ]

        self.triepilot_requested_draft_budgets = list(requested_draft_budgets)
        self.triepilot_bucketed_draft_budgets = list(bucketed_draft_budgets)
        self.triepilot_active_draft_lengths = list(active_draft_lengths)
        self.triepilot_bucket_ids = list(bucket_ids)
        self.triepilot_bucket_padding_nodes = list(bucket_padding_nodes)
        self.triepilot_shape_padding_tokens = list(shape_padding_tokens)
        self.triepilot_verify_draft_token_num = int(verify_draft_token_num)

        has_variable_budget = any(
            length != self.draft_token_num for length in active_draft_lengths
        )
        self.triepilot_structural_features = compute_ngram_tree_features(
            req_drafts=req_drafts,
            mask=mask,
            draft_token_num=self.draft_token_num,
            active_draft_lengths=active_draft_lengths,
        )
        if self.triepilot_shape_bucket_mode == "batch_max":
            fixed_inputs = self._build_fixed_shape_draft_inputs(
                batch,
                req_drafts,
                mask,
                active_draft_lengths,
                verify_draft_token_num,
            )
            draft_tokens = fixed_inputs["draft_tokens"]
            tree_mask = fixed_inputs["tree_mask"]
            positions = fixed_inputs["positions"]
            retrive_index = fixed_inputs["retrive_index"]
            retrive_next_token = fixed_inputs["retrive_next_token"]
            retrive_next_sibling = fixed_inputs["retrive_next_sibling"]
            draft_lens = None
        elif has_variable_budget:
            variable_inputs = self._build_variable_draft_inputs(
                batch, req_drafts, mask, active_draft_lengths
            )
            draft_tokens = torch.from_numpy(variable_inputs["draft_tokens"]).to(
                device=self.device, non_blocking=True
            )
            tree_mask = torch.from_numpy(variable_inputs["tree_mask"]).to(
                device=self.device, non_blocking=True
            )
            positions = torch.from_numpy(variable_inputs["positions"]).to(
                device=self.device, non_blocking=True
            )
            retrive_index = torch.from_numpy(variable_inputs["retrive_index"]).to(
                device=self.device, non_blocking=True
            )
            retrive_next_token = torch.from_numpy(
                variable_inputs["retrive_next_token"]
            ).to(device=self.device, non_blocking=True)
            retrive_next_sibling = torch.from_numpy(
                variable_inputs["retrive_next_sibling"]
            ).to(device=self.device, non_blocking=True)
            draft_lens = torch.tensor(
                active_draft_lengths, dtype=torch.int64, device=self.device
            )
        else:
            tree_mask.copy_(torch.from_numpy(mask), non_blocking=True)
            draft_tokens.copy_(torch.from_numpy(req_drafts), non_blocking=True)

            reconstruct_indices_from_tree_mask(
                tree_mask,
                batch.seq_lens,
                positions,  # mutable
                retrive_index,  # mutable
                retrive_next_token,  # mutable
                retrive_next_sibling,  # mutable
                bs,
                self.draft_token_num,
            )

            # NOTE: QLEN_MASK is faster than FULL_MASK, but requires corresponding changes in flashinfer.
            # Testing shows about 8% performance improvement (the effect is roughly proportional to batch size).
            if USE_FULL_MASK:
                tree_mask = []
                mask = mask.reshape(
                    batch.batch_size(), self.draft_token_num, self.draft_token_num
                )
                for i, req in enumerate(batch.reqs):
                    seq_len = len(req.origin_input_ids) + len(req.output_ids)
                    req_mask = torch.ones((self.draft_token_num, seq_len - 1)).cuda()
                    req_mask = torch.cat(
                        (req_mask, torch.from_numpy(mask[i]).cuda()), dim=1
                    ).to(torch.bool)
                    tree_mask.append(req_mask.flatten())
                tree_mask = torch.cat(tree_mask, dim=0)
            draft_lens = None

        batch.spec_algorithm = SpeculativeAlgorithm.NGRAM
        batch.forward_mode = ForwardMode.TARGET_VERIFY
        batch.spec_info = NgramVerifyInput(
            draft_tokens,
            tree_mask,
            positions,
            retrive_index,
            retrive_next_token,
            retrive_next_sibling,
            verify_draft_token_num,
            draft_lens=draft_lens,
            requested_draft_budgets=requested_draft_budgets,
            bucketed_draft_budgets=bucketed_draft_budgets,
            bucket_ids=bucket_ids,
            bucket_padding_nodes=bucket_padding_nodes,
            shape_padding_tokens=shape_padding_tokens,
        )
        batch.spec_info.prepare_for_verify(batch, self.page_size)

    def add_logprob_values(
        self,
        batch: ScheduleBatch,
        res: NgramVerifyInput,
        logits_output: LogitsProcessorOutput,
    ):
        # Extract args
        top_logprobs_nums = batch.top_logprobs_nums
        token_ids_logprobs = batch.token_ids_logprobs
        accepted_indices = res.accept_index
        assert len(accepted_indices) == len(logits_output.next_token_logits)

        temperatures = batch.sampling_info.temperatures
        num_draft_tokens = batch.spec_info.draft_token_num
        # acceptance indices are the indices in a "flattened" batch.
        # dividing it to num_draft_tokens will yield the actual batch index.
        temperatures = temperatures[accepted_indices // num_draft_tokens]
        if envs.SGLANG_RETURN_ORIGINAL_LOGPROB.get():
            logprobs = torch.nn.functional.log_softmax(
                logits_output.next_token_logits, dim=-1
            )
        else:
            logprobs = torch.nn.functional.log_softmax(
                logits_output.next_token_logits / temperatures, dim=-1
            )
        batch_next_token_ids = res.verified_id
        accept_length_per_req_cpu = res.accept_length.tolist()
        num_tokens_per_req = [accept + 1 for accept in accept_length_per_req_cpu]

        # We should repeat top_logprobs_nums to match num_tokens_per_req.
        top_logprobs_nums_repeat_interleaved = []
        token_ids_logprobs_repeat_interleaved = []
        for num, num_tokens in zip(top_logprobs_nums, num_tokens_per_req):
            top_logprobs_nums_repeat_interleaved.extend([num] * num_tokens)
        for token_ids, num_tokens in zip(token_ids_logprobs, num_tokens_per_req):
            token_ids_logprobs_repeat_interleaved.extend([token_ids] * num_tokens)

        # Extract logprobs
        if any(x > 0 for x in top_logprobs_nums):
            (
                logits_output.next_token_top_logprobs_val,
                logits_output.next_token_top_logprobs_idx,
            ) = get_top_logprobs(
                logprobs,
                top_logprobs_nums_repeat_interleaved,
            )

        if any(x is not None for x in token_ids_logprobs):
            (
                logits_output.next_token_token_ids_logprobs_val,
                logits_output.next_token_token_ids_logprobs_idx,
            ) = get_token_ids_logprobs(
                logprobs,
                token_ids_logprobs_repeat_interleaved,
            )

        logits_output.next_token_logprobs = logprobs[
            torch.arange(len(batch_next_token_ids), device=batch.sampling_info.device),
            batch_next_token_ids,
        ]

        # Add output logprobs to the request
        pt = 0
        next_token_logprobs = logits_output.next_token_logprobs.tolist()
        verified_ids = batch_next_token_ids.tolist()
        for req, num_tokens in zip(batch.reqs, num_tokens_per_req, strict=True):
            for _ in range(num_tokens):
                if req.return_logprob:
                    req.output_token_logprobs_val.append(next_token_logprobs[pt])
                    req.output_token_logprobs_idx.append(verified_ids[pt])
                    if req.top_logprobs_num > 0:
                        req.output_top_logprobs_val.append(
                            logits_output.next_token_top_logprobs_val[pt]
                        )
                        req.output_top_logprobs_idx.append(
                            logits_output.next_token_top_logprobs_idx[pt]
                        )
                pt += 1

    def _update_ngram_cache(self, batch: ScheduleBatch):
        batch_tokens = []
        for req in batch.reqs:
            # FIXME: Whether to insert 'extend' into the cache or not, after testing,
            # there is not much difference, so we will not insert it for now.
            # if batch.forward_mode.is_extend():
            #     put_ids = req.origin_input_ids + req.output_ids
            # else:
            put_ids = self._efficient_concat_last_n(
                req.origin_input_ids, req.output_ids, self.branch_length
            )
            batch_tokens.append(put_ids)
        self.ngram_cache.batch_put(batch_tokens)

    def forward_batch_generation(self, batch: ScheduleBatch) -> GenerationBatchResult:
        step_start_ns = time.perf_counter_ns()
        self._prepare_for_speculative_decoding(batch)
        prepare_done_ns = time.perf_counter_ns()
        model_worker_batch = batch.get_model_worker_batch()
        num_accepted_tokens = 0
        accept_lens = None

        if model_worker_batch.forward_mode.is_target_verify():
            batch_result = self.target_worker.forward_batch_generation(
                model_worker_batch, is_verify=True
            )
            target_forward_done_ns = time.perf_counter_ns()
            logits_output, can_run_cuda_graph = (
                batch_result.logits_output,
                batch_result.can_run_cuda_graph,
            )
            verify_input = model_worker_batch.spec_info
            logits_output, next_token_ids, num_accepted_tokens = verify_input.verify(
                batch, logits_output, self.page_size
            )
            verify_done_ns = time.perf_counter_ns()
            # Store accept_lens for per-request metrics
            accept_lens = verify_input.accept_length
            accept_len_emas = observe_triepilot_accept_lengths(
                batch.reqs,
                accept_lens,
                alpha=self.triepilot_accept_ema_alpha,
            )
            observe_triepilot_strategy_feedback(
                batch.reqs,
                strategy_bank=self.triepilot_strategy_bank,
                allocation_metadata=self.triepilot_allocation_metadata,
                requested_draft_budgets=getattr(
                    verify_input, "requested_draft_budgets", None
                ),
                accept_lens=accept_lens,
                alpha=self.triepilot_accept_ema_alpha,
                step_id=self.triepilot_step_id,
            )
            if batch.return_logprob:
                self.add_logprob_values(batch, verify_input, logits_output)
            self._update_ngram_cache(batch)
            batch.forward_mode = ForwardMode.DECODE
            if self.triepilot_recorder.enabled:
                self.triepilot_recorder.record_step(
                    step_id=self.triepilot_step_id,
                    batch_size=batch.batch_size(),
                    request_ids=[req.rid for req in batch.reqs],
                    seq_lens=batch.seq_lens_cpu,
                    draft_token_num=self.draft_token_num,
                    verify_draft_token_num=getattr(
                        verify_input, "draft_token_num", self.draft_token_num
                    ),
                    requested_draft_budgets=getattr(
                        verify_input, "requested_draft_budgets", None
                    ),
                    bucketed_draft_budgets=getattr(
                        verify_input, "bucketed_draft_budgets", None
                    ),
                    active_draft_lengths=(
                        self.triepilot_active_draft_lengths
                        or getattr(verify_input, "draft_lens_cpu", None)
                    ),
                    bucket_ids=getattr(verify_input, "bucket_ids", None),
                    bucket_padding_nodes=getattr(
                        verify_input, "bucket_padding_nodes", None
                    ),
                    shape_padding_tokens=getattr(
                        verify_input, "shape_padding_tokens", None
                    ),
                    accept_lens=accept_lens,
                    num_accepted_tokens=num_accepted_tokens,
                    can_run_cuda_graph=can_run_cuda_graph,
                    timings_ns={
                        "ngram_query": prepare_done_ns - step_start_ns,
                        "target_forward": target_forward_done_ns - prepare_done_ns,
                        "verify": verify_done_ns - target_forward_done_ns,
                        "step": verify_done_ns - step_start_ns,
                    },
                    structural_features=self.triepilot_structural_features,
                    allocation_policy=self.triepilot_allocation_policy,
                    batch_budget=self.triepilot_batch_budget,
                    accept_len_emas=accept_len_emas,
                    allocation_metadata=self.triepilot_allocation_metadata,
                    controller_time_ns=self.triepilot_controller_time_ns,
                )
                self.triepilot_step_id += 1

        else:
            batch_result = self.target_worker.forward_batch_generation(
                model_worker_batch
            )
            logits_output, next_token_ids, can_run_cuda_graph = (
                batch_result.logits_output,
                batch_result.next_token_ids,
                batch_result.can_run_cuda_graph,
            )

        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=next_token_ids,
            num_accepted_tokens=num_accepted_tokens,
            can_run_cuda_graph=can_run_cuda_graph,
            accept_lens=accept_lens,
        )
