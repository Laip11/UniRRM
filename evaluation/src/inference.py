"""Inference engines for reward-model evaluation.

All engines work with *n* candidate responses (n=2 for pairwise, n=4 for
listwise, or any other value).  The caller provides answer columns via
``answer_keys`` -- e.g. ``["answer_a", "answer_b"]`` for pairwise or
``["answer_a", "answer_b", "answer_c", "answer_d"]`` for listwise.

Engine families (all under the GenRM umbrella)
----------------------------------------------
- **GenRMJudgeEngine** (``genrm-judge``) -- all *n* responses in one prompt; the LLM directly outputs the winner.

- **GenRMScorerEngine** (``genrm-scorer``) -- each response is scored independently.

- **ScalarRMEngine** (``scalar-rm``) -- a local scalar reward model scores each response; the highest-scoring response wins.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

import torch
from tqdm import tqdm, trange
from vllm import LLM

from src.utils import generate_with_vllm

INDEX_TO_LETTER = {0: "A", 1: "B", 2: "C", 3: "D"}

PAIRWISE_ANSWER_KEYS = ["answer_a", "answer_b"]
LISTWISE_ANSWER_KEYS = ["answer_a", "answer_b", "answer_c", "answer_d"]

DEFAULT_GPU_MEMORY_UTILIZATION = 0.8


def _format_verdict(best_idx: int, num_candidates: int) -> str:
    """Build a verdict string consistent with existing parsers."""
    letter = INDEX_TO_LETTER[best_idx]
    if num_candidates == 2:
        return f"[[{letter}]]"
    return f"Verdict:[[{letter}]]"


class _PointwisePromptValues(dict):
    """Keep unknown template placeholders untouched during formatting."""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def _format_pointwise_prompt(prompt_template: str, question: str, answer: str) -> str:
    """Format pointwise prompts with common placeholder aliases."""
    template_values = _PointwisePromptValues({
        "question": question,
        "answer": answer,
    })
    return prompt_template.format_map(template_values)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseInferenceEngine(ABC):
    """Common interface for all inference strategies.

    Parameters
    ----------
    answer_keys : list[str]
        Column names in ``eval_data`` that contain the candidate responses.
        Length determines *n* (number of candidates).
    """

    def __init__(self, answer_keys: List[str]):
        self.answer_keys = answer_keys
        self.num_candidates = len(answer_keys)

    @abstractmethod
    def infer(
        self,
        eval_data,
        model_template,
        need_apply_chat_template: bool,
        args,
    ) -> List[Dict[str, Any]]:
        """Return a list of ``{"response": str, ...}`` dicts, one per sample."""


# ---------------------------------------------------------------------------
# GenRMJudgeEngine (genrm-judge)
# All n responses in one prompt; LLM directly picks the winner.
# ---------------------------------------------------------------------------

class GenRMJudgeEngine(BaseInferenceEngine):
    """Submit pre-built judge prompts (``user_prompt`` column) to vLLM.

    Works for any *n*: the prompt template already contains all candidates.
    n=2 is pairwise judging, n>2 is listwise judging.
    """

    def __init__(
        self,
        model_name_or_path: str,
        answer_keys: List[str],
        gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION,
        tensor_parallel_size: Optional[int] = None,
    ):
        super().__init__(answer_keys)
        self.llm = LLM(
            model=model_name_or_path,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
            tensor_parallel_size=tensor_parallel_size or torch.cuda.device_count(),
        )
        self.tokenizer = self.llm.get_tokenizer()

    def infer(self, eval_data, model_template, need_apply_chat_template, args):
        prompts = eval_data["user_prompt"]
        results = generate_with_vllm(
            llm=self.llm,
            prompts=prompts,
            tokenizer=self.tokenizer,
            need_apply_chat_template=need_apply_chat_template,
            system_prompt=model_template.system_template_list,
            enable_thinking=model_template.enable_thinking,
            max_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        return results


# ---------------------------------------------------------------------------
# GenRMScorerEngine (genrm-scorer)
# Score each of n responses independently, pick best.
# ---------------------------------------------------------------------------


class GenRMScorerEngine(BaseInferenceEngine):
    """Score each of *n* answers independently via pointwise evaluation.

    Returns raw model responses for each candidate so that the result
    calculator can extract scores and determine the winner.

    Works for any ``n``: pairwise (n=2), listwise (n=4), or beyond.
    """

    def __init__(
        self,
        model_name_or_path: str,
        answer_keys: List[str],
        gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION,
        tensor_parallel_size: Optional[int] = None,
    ):
        super().__init__(answer_keys)
        self.llm = LLM(
            model=model_name_or_path,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
            tensor_parallel_size=tensor_parallel_size or torch.cuda.device_count(),
        )
        self.tokenizer = self.llm.get_tokenizer()

    def infer(self, eval_data, model_template, need_apply_chat_template, args):
        prompts_ls = eval_data["prompt"]
        answer_columns = [eval_data[key] for key in self.answer_keys]
        pointwise_prompt_template = model_template.user_template_point

        results = []
        for idx in tqdm(range(len(prompts_ls))):
            prompt = prompts_ls[idx]
            input_list = [
                _format_pointwise_prompt(
                    pointwise_prompt_template,
                    question=prompt,
                    answer=answer_columns[k][idx],
                )
                for k in range(self.num_candidates)
            ]
            response_list = generate_with_vllm(
                llm=self.llm,
                prompts=input_list,
                tokenizer=self.tokenizer,
                need_apply_chat_template=need_apply_chat_template,
                system_prompt=model_template.system_template_point,
                enable_thinking=model_template.enable_thinking,
                max_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
            )

            results.append({
                "responses": [r["response"] for r in response_list],
            })
        return results


# ---------------------------------------------------------------------------
# ScalarRMEngine (scalar-rm)
# Local scalar reward model via vLLM pooling (Skywork-Reward style).
# ---------------------------------------------------------------------------

class ScalarRMEngine(BaseInferenceEngine):
    """Score *n* candidate responses using a local scalar reward model.

    Uses vLLM's pooling runner to load sequence-classification reward models
    (e.g. Skywork-Reward-V2-Llama-3.1-8B) and score each conversation locally.

    Works for any ``n``: pairwise (n=2), listwise (n=4), or beyond.
    """

    def __init__(
        self,
        model_name_or_path: str,
        answer_keys: List[str],
        gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION,
        tensor_parallel_size: Optional[int] = None,
    ):
        super().__init__(answer_keys)
        self.llm = LLM(
            model=model_name_or_path,
            runner="pooling",
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
            tensor_parallel_size=tensor_parallel_size or torch.cuda.device_count(),
        )
        self.tokenizer = self.llm.get_tokenizer()

    def infer(self, eval_data, model_template, need_apply_chat_template, args):
        prompts_ls = eval_data["prompt"]
        answer_columns = [eval_data[key] for key in self.answer_keys]

        # Build all conversations: for each sample, format n candidate conversations
        all_conversations = []
        for row_idx in range(len(prompts_ls)):
            for col_idx in range(self.num_candidates):
                conv = [
                    {"role": "user", "content": prompts_ls[row_idx]},
                    {"role": "assistant", "content": answer_columns[col_idx][row_idx]},
                ]
                formatted = self.tokenizer.apply_chat_template(conv, tokenize=False)
                all_conversations.append(formatted)

        # Score all conversations using vLLM pooling
        outputs = self.llm.encode(all_conversations, pooling_task="classify")
        all_scores = [output.outputs.data[0] for output in outputs]

        # Pick the best candidate for each sample
        results = []
        for row_idx in range(len(prompts_ls)):
            best_idx = 0
            best_score = float("-inf")
            for col_idx in range(self.num_candidates):
                score = all_scores[row_idx * self.num_candidates + col_idx]
                if score > best_score:
                    best_score = score
                    best_idx = col_idx

            results.append({
                "response": _format_verdict(best_idx, self.num_candidates),
            })
        return results


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_engine(
    reward_type: str,
    model_name_or_path: str,
    answer_keys: Optional[List[str]] = None,
    gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION,
    tensor_parallel_size: Optional[int] = None,
) -> BaseInferenceEngine:
    """Instantiate the appropriate inference engine.

    Parameters
    ----------
    reward_type : str
        One of ``"genrm-judge"``, ``"genrm-scorer"``, ``"scalar-rm"``.
    model_name_or_path : str
        HuggingFace model id or local path.
    answer_keys : list[str], optional
        Column names for candidate responses.  Defaults to pairwise
        (``["answer_a", "answer_b"]``).  Pass 4 keys for listwise, etc.
    gpu_memory_utilization : float
        Fraction of GPU memory used by vLLM engines.
    tensor_parallel_size : int, optional
        Tensor parallel size for vLLM engines. Defaults to all visible GPUs.
    """
    if answer_keys is None:
        answer_keys = PAIRWISE_ANSWER_KEYS

    if reward_type == "genrm-judge":
        return GenRMJudgeEngine(
            model_name_or_path,
            answer_keys,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
        )

    if reward_type == "genrm-scorer":
        return GenRMScorerEngine(
            model_name_or_path,
            answer_keys,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
        )

    if reward_type == "scalar-rm":
        return ScalarRMEngine(
            model_name_or_path,
            answer_keys,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
        )

    raise ValueError(
        f"Unknown reward_type: {reward_type!r}. "
        f"Choose from: 'genrm-judge', 'genrm-scorer', 'scalar-rm'."
    )
