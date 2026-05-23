"""Evaluation prompt templates package.

Provides :data:`TEMPLATE_REGISTRY` — a mapping from template name strings
to :class:`EvalPromptTemplate` instances — and re-exports all public helpers
from :mod:`._base`.
"""

from ._base import (
    EvalPromptTemplate,
    extract_score_json_pointwise,
    extract_winner_best_id_or_scores,
    extract_winner_by_score_comparison,
    extract_winner_json_listwise,
    extract_winner_json_pairwise,
    extract_winner_xml_json,
    make_extractor,
    make_pointwise_extractor,
    regex_extractor,
)

from .scalar_rm import default_scalar_rm
from .skywork_critic import Skywork_Critic_Llama_3_1_70B
from .rm_r1_qwen import RM_R1_Qwen2_5_Instruct_32B
from .nemotron import Llama_3_3_Nemotron_Super_49B_GenRM_Multilingual
from .mr3_genrm import mR3_GenRM
from .m_prometheus import m_prometheus
from .judge_lrm import JudgeLRM
from .openrubric import openrubric
from .unirrm import UniRRM

TEMPLATE_REGISTRY = {
    "scalar_rm": default_scalar_rm,
    "RM_R1_Qwen2.5_Instruct_32B": RM_R1_Qwen2_5_Instruct_32B,
    "Skywork_Critic_Llama_3_1_70B": Skywork_Critic_Llama_3_1_70B,
    "Llama_3_3_Nemotron_Super_49B_GenRM_Multilingual": Llama_3_3_Nemotron_Super_49B_GenRM_Multilingual,
    "mR3_GenRM": mR3_GenRM,
    "m-prometheus": m_prometheus,
    "JudgeLRM": JudgeLRM,
    "openrubric": openrubric,
    "UniRRM": UniRRM,
}

__all__ = [
    "EvalPromptTemplate",
    "TEMPLATE_REGISTRY",
    "extract_score_json_pointwise",
    "extract_winner_best_id_or_scores",
    "extract_winner_by_score_comparison",
    "extract_winner_json_listwise",
    "extract_winner_json_pairwise",
    "extract_winner_xml_json",
    "make_extractor",
    "make_pointwise_extractor",
    "regex_extractor",
    "default_scalar_rm",
    "Skywork_Critic_Llama_3_1_70B",
    "RM_R1_Qwen2_5_Instruct_32B",
    "Llama_3_3_Nemotron_Super_49B_GenRM_Multilingual",
    "mR3_GenRM",
    "m_prometheus",
    "JudgeLRM",
    "openrubric",
    "UniRRM",
]
