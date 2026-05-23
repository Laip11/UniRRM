import re
from typing import Any, Dict, List, Optional, Tuple

BOX_WIDTH = 58

INDEX_TO_LETTER = {0: "A", 1: "B", 2: "C", 3: "D"}

_VERDICT_PATTERN = re.compile(r"\[\[([A-D])\]\]")


def _pick_winner_from_pointwise(responses: List[str], extractor) -> Optional[str]:
    """Extract scores from pointwise responses and return the winner letter.

    Parameters
    ----------
    responses : list[str]
        Raw model outputs for each candidate (one per response).
    extractor : callable
        ``pointwise_answer_extractor(text) -> Optional[float]``

    Returns
    -------
    str or None
        Winner letter ("A", "B", ...) or None if all extractions fail.
    """
    best_idx = None
    best_score = float("-inf")
    all_failed = True

    for idx, response_text in enumerate(responses):
        raw_score = extractor(response_text)
        if raw_score is None:
            continue
        all_failed = False
        score = float(raw_score)
        if score > best_score:
            best_score = score
            best_idx = idx

    if all_failed or best_idx is None:
        return None
    return INDEX_TO_LETTER[best_idx]


def _box_line(content: str) -> str:
    """Wrap content in box-drawing side borders, padded to BOX_WIDTH."""
    inner = content.ljust(BOX_WIDTH - 4)
    return f"\u2502 {inner} \u2502\n"


def _box_top() -> str:
    return "\u250c" + "\u2500" * (BOX_WIDTH - 2) + "\u2510\n"


def _box_mid() -> str:
    return "\u251c" + "\u2500" * (BOX_WIDTH - 2) + "\u2524\n"


def _box_bottom() -> str:
    return "\u2514" + "\u2500" * (BOX_WIDTH - 2) + "\u2518\n"


def compute_pairwise_accuracy(
    category_datasets: Dict[str, Any],
    category_results: Dict[str, List[Dict[str, Any]]],
    model_template,
) -> Tuple[Dict[str, Dict[str, Any]], int]:
    """Compute per-category accuracy for pairwise evaluation.

    Parameters
    ----------
    category_datasets : dict
        Mapping of category name to dataset (each row has a ``winner`` field).
    category_results : dict
        Mapping of category name to list of inference result dicts.
        For judge engines: each dict has ``{"response": str}``.
        For scorer engines: each dict has ``{"responses": list[str]}``.
    model_template :
        Template object providing ``listwise_answer_extractor`` and
        ``pointwise_answer_extractor``.

    Returns
    -------
    category_accuracy : dict
        ``{category: {"accuracy": float|None, "total": int, "invalid": int}}``
    total_missed : int
        Total number of parse failures across all categories.
    """
    category_accuracy: Dict[str, Dict[str, Any]] = {}
    total_missed = 0

    for category, dataset in category_datasets.items():
        results = category_results[category]
        winners = dataset["winner"]
        total_count = len(winners)
        correct = 0
        invalid_count = 0

        for idx in range(total_count):
            result = results[idx]
            if "responses" in result:
                predicted = _pick_winner_from_pointwise(
                    result["responses"], model_template.pointwise_answer_extractor
                )
            else:
                response_text = result["response"]
                predicted = model_template.listwise_answer_extractor(response_text)
                if predicted is None:
                    match = _VERDICT_PATTERN.search(response_text)
                    predicted = match.group(1) if match else None
            if predicted is None:
                invalid_count += 1
                continue
            if predicted == winners[idx]:
                correct += 1

        valid_count = total_count - invalid_count
        total_missed += invalid_count

        if valid_count == 0:
            category_accuracy[category] = {
                "accuracy": None,
                "total": total_count,
                "invalid": invalid_count,
            }
        else:
            category_accuracy[category] = {
                "accuracy": correct / valid_count,
                "total": total_count,
                "invalid": invalid_count,
            }

    return category_accuracy, total_missed


def compute_listwise_accuracy(
    eval_data,
    results: List[Dict[str, Any]],
    model_template,
) -> Tuple[float, int, int]:
    """Compute accuracy for listwise (1-of-N) evaluation.

    Parameters
    ----------
    eval_data :
        Dataset with a ``winner`` column.
    results : list[dict]
        Inference results. For judge engines: ``{"response": str}``.
        For scorer engines: ``{"responses": list[str]}``.
    model_template :
        Template object providing ``listwise_answer_extractor`` and
        ``pointwise_answer_extractor``.

    Returns
    -------
    accuracy : float
        Fraction of correct predictions among valid ones.
    invalid_count : int
        Number of parse failures.
    total_len : int
        Total number of samples.
    """
    winners = eval_data["winner"]
    total_len = len(winners)
    correct = 0
    invalid_count = 0

    for idx in range(total_len):
        result = results[idx]
        if "responses" in result:
            predicted = _pick_winner_from_pointwise(
                result["responses"], model_template.pointwise_answer_extractor
            )
        else:
            response_text = result["response"]
            predicted = model_template.listwise_answer_extractor(response_text)
            if predicted is None:
                match = _VERDICT_PATTERN.search(response_text)
                predicted = match.group(1) if match else None
        if predicted is None:
            invalid_count += 1
            continue
        if predicted == winners[idx]:
            correct += 1

    valid_count = total_len - invalid_count
    accuracy = correct / valid_count if valid_count > 0 else 0.0
    return accuracy, invalid_count, total_len


def save_pairwise_report(
    log_file: str,
    data_name: str,
    category_accuracy: Dict[str, Dict[str, Any]],
    missed_count: int,
) -> None:
    """Append an evaluation report with per-category breakdown to ``log_file``."""
    total_len = sum(
        int(result["total"] or 0) for result in category_accuracy.values()
    )

    with open(log_file, "a", encoding="utf-8") as file_handle:
        file_handle.write("\n")
        file_handle.write(_box_top())
        file_handle.write(_box_line(f"Data Name      : {data_name}"))
        file_handle.write(_box_line(f"Total samples  : {total_len}"))
        file_handle.write(_box_line(f"Parse failures : {missed_count}"))
        file_handle.write(_box_mid())

        header = f"{'Category':<16}{'Accuracy':>10}{'Count':>10}{'Failures':>12}"
        file_handle.write(_box_line(header))
        separator = "\u2500" * (BOX_WIDTH - 4)
        file_handle.write(_box_line(separator))

        for category, result in category_accuracy.items():
            cat_accuracy = result["accuracy"]
            total_count = int(result["total"] or 0)
            cat_invalid = int(result["invalid"] or 0)
            accuracy_text = "N/A" if cat_accuracy is None else f"{float(cat_accuracy):.4f}"
            row = f"{category:<16}{accuracy_text:>10}{total_count:>10}{cat_invalid:>12}"
            file_handle.write(_box_line(row))

        file_handle.write(_box_mid())

        if total_len == 0:
            average_accuracy = 0.0
        else:
            total_valid = sum(
                int(result["total"] or 0) - int(result["invalid"] or 0)
                for result in category_accuracy.values()
            )
            if total_valid == 0:
                average_accuracy = 0.0
            else:
                weighted_correct = sum(
                    float(result["accuracy"] or 0) * (int(result["total"] or 0) - int(result["invalid"] or 0))
                    for result in category_accuracy.values()
                    if result["accuracy"] is not None
                )
                average_accuracy = weighted_correct / total_valid

        avg_row = f"{'Average':<16}{average_accuracy:>10.4f}{total_len:>10}{missed_count:>12}"
        file_handle.write(_box_line(avg_row))
        file_handle.write(_box_bottom())
        file_handle.write("\n")


def save_listwise_report(
    log_file: str,
    data_name: str,
    accuracy: float,
    invalid_count: int,
    total_len: int,
) -> None:
    """Append an evaluation report to ``log_file``."""
    with open(log_file, "a", encoding="utf-8") as file_handle:
        file_handle.write("\n")
        file_handle.write(_box_top())
        file_handle.write(_box_line(f"Data Name      : {data_name}"))
        file_handle.write(_box_line(f"Total samples  : {total_len}"))
        file_handle.write(_box_line(f"Parse failures : {invalid_count}"))
        file_handle.write(_box_mid())
        file_handle.write(_box_line(f"{'Accuracy':<16}{accuracy:>10.4f}"))
        file_handle.write(_box_bottom())
        file_handle.write("\n")