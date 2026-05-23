"""Base module for evaluation prompt templates.

Contains:
- Built-in answer extractor helpers
- Extractor factories
- The ``EvalPromptTemplate`` dataclass

Field naming conventions
------------------------
- **system_template_point**: System prompt for pointwise evaluation.
- **system_template_list**: System prompt for listwise evaluation (pair is n=2 case).
- **user_template_point**: User-message template for pointwise evaluation.
- **user_template_list**: User-message template for listwise/pairwise evaluation.
- **formatted_prompt**: A pre-formatted prompt string that already has chat
  template applied (e.g. with special tokens baked in). When set, the engine
  skips ``apply_chat_template`` and uses this directly.
- **enable_thinking**: Whether to enable chain-of-thought in the model output.

Extractor protocol
------------------
- **listwise_answer_extractor(text: str) -> Optional[str]**
  Given the raw model response, return the winner label (``"A"`` / ``"B"`` /
  ``"C"`` / ``"D"``) or ``None`` on failure.
- **pointwise_answer_extractor(text: str) -> Optional[str | int | float]**
  Given the raw model response, return the scalar score or ``None``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable, Optional, Union

# ---------------------------------------------------------------------------
# Built-in answer extractor helpers
# ---------------------------------------------------------------------------


def regex_extractor(
    text: str,
    pattern: str,
    answer_type: type = str,
) -> Optional[str]:
    """Extract a winner from *text* using a regex pattern.

    When *answer_type* is ``int`` the regex is expected to capture two
    numeric groups which are compared to decide A vs B.
    """
    if not text or not isinstance(text, str):
        return None
    match = re.findall(pattern, text)
    if not match:
        return None
    if answer_type == int:
        score_pair = match[0]
        if len(score_pair) == 2:
            score_a, score_b = int(score_pair[0]), int(score_pair[1])
            return "A" if score_a > score_b else "B"
        return None
    return match[0]


def _extract_json_string(text: str) -> Optional[str]:
    """Pull a JSON object out of *text* (code-fenced or bare)."""
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.S)
    if fenced:
        return fenced.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1:
        return text[start : end + 1]
    return None


def _normalize_id(raw_id, mapping: dict) -> Optional[str]:
    if not raw_id:
        return None
    string = str(raw_id).strip().upper()
    if string in mapping:
        return mapping[string]
    for digit, letter in mapping.items():
        if digit in string:
            return letter
    return string if string in set(mapping.values()) else None


def _winner_from_json_data(
    data: dict,
    mapping: dict,
) -> Optional[str]:
    """Try ``best_id`` first, then fall back to comparing ``final_score``."""
    if "best_id" in data and data["best_id"]:
        result = _normalize_id(data["best_id"], mapping)
        if result:
            return result

    evals = data.get("evaluations", [])
    if not isinstance(evals, list) or len(evals) == 0:
        return None
    best_id, max_score = None, -1.0
    for item in evals:
        if not isinstance(item, dict):
            continue
        try:
            score = float(item.get("final_score", -1))
            rid = _normalize_id(item.get("response_id"), mapping)
        except (TypeError, ValueError):
            continue
        if score > max_score and rid:
            max_score = score
            best_id = rid
    return best_id


def extract_winner_json_pairwise(text: str) -> Optional[str]:
    """Extract winner (A/B) from a JSON response with ``best_id`` / ``evaluations``."""
    if not text:
        return None
    json_str = _extract_json_string(text)
    if json_str is None:
        return None
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return None
    return _winner_from_json_data(data, {"1": "A", "2": "B"})


def extract_winner_json_listwise(text: str) -> Optional[str]:
    """Extract winner (A/B/C/D) from a JSON response."""
    if not text:
        return None
    json_str = _extract_json_string(text)
    if json_str is None:
        return None
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return None
    return _winner_from_json_data(data, {"1": "A", "2": "B", "3": "C", "4": "D"})


def extract_winner_xml_json(text: str) -> Optional[str]:
    """Extract winner from ``<results>`` XML-wrapped JSON (legacy ``json`` mode)."""
    if not text:
        return None
    match = re.search(r"<results>\s*(.*?)\s*</results>", text, re.S)
    if not match:
        return None
    content = re.sub(r"```json|```", "", match.group(1)).strip()
    try:
        data = json.loads(content)
        if data[0]["final_score"] > data[1]["final_score"]:
            return "A"
        return "B"
    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
        return None


def extract_winner_best_id_or_scores(text: str) -> Optional[str]:
    """Extract winner via ``best_id`` regex or by comparing ``final_score`` values."""
    if not text:
        return None
    best_id_match = re.search(r'"best_id"\s*:\s*"Response([12])"', text)
    if best_id_match:
        return "A" if best_id_match.group(1) == "1" else "B"

    scores = re.findall(r'"final_score"\s*:\s*([0-9]+(?:\.[0-9]+)?)', text)
    if len(scores) < 2:
        return None
    score_a, score_b = float(scores[-2]), float(scores[-1])
    if score_a > score_b:
        return "A"
    elif score_a < score_b:
        return "B"
    return None


def extract_score_json_pointwise(text: str) -> Optional[float]:
    """Extract a single pointwise score from a JSON response.

    Parses the ``evaluations[0].final_score`` field from JSON that may be
    code-fenced or bare.  Falls back to regex if JSON parsing fails.
    """
    if not text:
        return None
    text = text.split("</think>")[-1]
    json_str = _extract_json_string(text)
    if json_str:
        try:
            data = json.loads(json_str)
            if isinstance(data, dict) and "evaluations" in data:
                evals = data["evaluations"]
                if isinstance(evals, list) and len(evals) > 0:
                    score = evals[0].get("final_score")
                    if score is not None:
                        return float(score)
            elif isinstance(data, list) and len(data) > 0:
                score = data[0].get("final_score")
                if score is not None:
                    return float(score)
        except (json.JSONDecodeError, TypeError, ValueError, AttributeError):
            pass

    score_match = re.search(r'"final_score"\s*:\s*(-?\d+(?:\.\d+)?)', text)
    if score_match:
        return float(score_match.group(1))
    return None


def extract_winner_by_score_comparison(
    text: str,
    pattern: str,
) -> Optional[str]:
    """Extract two numeric scores via *pattern* and return the winner by comparison.

    The regex must capture exactly two groups (score_a, score_b).
    Returns ``"A"`` if score_a > score_b, ``"B"`` otherwise.
    """
    if not text:
        return None
    match = re.findall(pattern, text)
    if not match:
        return None
    score_pair = match[0]
    if len(score_pair) != 2:
        return None
    try:
        score_a, score_b = float(score_pair[0]), float(score_pair[1])
    except (TypeError, ValueError):
        return None
    return "A" if score_a > score_b else "B"


# ---------------------------------------------------------------------------
# Extractor factories — convenience wrappers for common patterns
# ---------------------------------------------------------------------------


def make_extractor(
    pattern: str,
    answer_type: type = str,
) -> Callable[[str], Optional[str]]:
    """Create a listwise answer extractor from a regex pattern.

    - When *answer_type* is ``str``, the first capture group is returned directly.
    - When *answer_type* is ``int``, the regex must capture two numeric groups
      which are compared to decide A vs B (delegates to
      :func:`extract_winner_by_score_comparison`).

    For non-regex extraction, pass any ``(str) -> Optional[str]`` callable
    directly to ``listwise_answer_extractor`` instead of using this factory.
    """
    if answer_type == int:
        def _score_extractor(text: str) -> Optional[str]:
            return extract_winner_by_score_comparison(text, pattern)
        return _score_extractor

    def _regex_extractor(text: str) -> Optional[str]:
        return regex_extractor(text, pattern, str)
    return _regex_extractor


def make_pointwise_extractor(
    pattern: str,
) -> Callable[[str], Optional[Union[str, int, float]]]:
    """Create a pointwise score extractor from a regex pattern.

    The first capture group is returned as a string.  For structured
    extraction (e.g. JSON), pass a custom callable directly to
    ``pointwise_answer_extractor`` instead.
    """
    def _extractor(text: str) -> Optional[str]:
        if not text:
            return None
        match = re.findall(pattern, text)
        return match[0] if match else None
    return _extractor


# ---------------------------------------------------------------------------
# Template dataclass
# ---------------------------------------------------------------------------


@dataclass
class EvalPromptTemplate:
    """Prompt template for an evaluation model.

    Fields
    ------
    template_name : str
        Unique identifier for the template.
    system_template_point : str, optional
        System prompt for pointwise evaluation.
    system_template_list : str, optional
        System prompt for listwise/pairwise evaluation.
    user_template_point : str, optional
        User-message template for pointwise evaluation.
    user_template_list : str, optional
        User-message template for listwise/pairwise evaluation.
    formatted_prompt : str, optional
        A pre-formatted prompt string with chat template already applied.
        When set, the engine skips ``apply_chat_template``.
    enable_thinking : bool
        Whether to enable chain-of-thought reasoning.
    listwise_answer_extractor : callable, optional
        ``(text: str) -> Optional[str]``  Custom listwise winner parser.
    pointwise_answer_extractor : callable, optional
        ``(text: str) -> Optional[str|int|float]``  Custom pointwise score parser.
    """

    template_name: Optional[str] = None
    system_template_point: Optional[str] = None
    system_template_list: Optional[str] = None
    user_template_list: Optional[str] = None
    user_template_point: Optional[str] = None
    formatted_prompt: Optional[str] = None
    enable_thinking: bool = False

    # -- Pluggable answer extraction callbacks -------------------------------
    listwise_answer_extractor: Optional[Callable[[str], Optional[str]]] = field(
        default=None, repr=False,
    )
    pointwise_answer_extractor: Optional[Callable[[str], Optional[Union[str, int, float]]]] = field(
        default=None, repr=False,
    )
