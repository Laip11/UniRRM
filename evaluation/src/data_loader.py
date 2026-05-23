"""Data loading utilities for pairwise and listwise evaluation."""

import numpy as np
from datasets import Dataset, DatasetDict, load_dataset, concatenate_datasets
np.random.seed(42)
default_cache_dir = "/mnt/workspace/laip/data"

# ---------------------------------------------------------------------------
# Pairwise data loading
# ---------------------------------------------------------------------------

CATEGORY_SOURCES = {
    "Reasoning": [
        "hep-python", "hep-go", "hep-cpp", "hep-js", "hep-rust",
        "hep-java", "math-prm",
    ],
    "Chat": [
        "alpacaeval-easy", "alpacaeval-hard", "alpacaeval-length",
        "mt-bench-easy", "mt-bench-med",
    ],
    "Chat Hard": [
        "mt-bench-hard", "llmbar-natural", "llmbar-adver-neighbor",
        "llmbar-adver-GPTInst", "llmbar-adver-GPTOut",
        "llmbar-adver-manual",
    ],
    "Safety": [
        "refusals-dangerous", "refusals-offensive",
        "xstest-should-respond", "xstest-should-refuse", "donotanswer",
    ],
}


def load_pairwise_dataset(
    data_name: str,
    cache_dir: str = default_cache_dir,
) -> Dataset:

    if data_name == "rewardbench":
        dataset = load_dataset(
            "allenai/reward-bench", split="filtered", cache_dir=cache_dir
        )
        source_to_category = {
            src: cat
            for cat, sources in CATEGORY_SOURCES.items()
            for src in sources
        }
        dataset = dataset.map(
            lambda ex: {"category": source_to_category.get(ex.get("subset", ""), "Unknown")}
        )

    elif data_name == "MM-Eval":
        dataset = load_dataset(
            "prometheus-eval/MM-Eval", split="test", cache_dir=cache_dir
        )
        dataset = dataset.add_column("category", dataset["subset"])
        dataset = dataset.filter(lambda x: "lang_res" not in x["category"])

    elif data_name == "judgebench":
        raw = load_dataset("ScalerLab/JudgeBench", cache_dir=cache_dir)
        merged = concatenate_datasets([raw["gpt"], raw["claude"]])

        def _convert_judgebench(example):
            if example["label"] == "A>B":
                chosen, rejected = example["response_A"], example["response_B"]
            else:
                chosen, rejected = example["response_B"], example["response_A"]
            return {"prompt": example["question"], "chosen": chosen, "rejected": rejected}

        dataset = merged.map(_convert_judgebench).remove_columns(
            ["response_model", "response_A", "response_B", "label", "original_id", "pair_id", "source"]
        )

    elif data_name == "judgebench_pro":
        dataset = load_dataset(
            "SUSTech-NLP/JudgeBench-Pro",
            split="test",
            cache_dir=cache_dir,
            trust_remote_code=True,
        )

        def _convert_judgebench_pro(example):
            if example["label"] == 1:
                chosen, rejected = example["response_A"], example["response_B"]
            else:
                chosen, rejected = example["response_B"], example["response_A"]
            return {
                "prompt": example["question"],
                "chosen": chosen,
                "rejected": rejected,
                "category": example.get("domain", "all"),
            }

        dataset = dataset.map(
            _convert_judgebench_pro,
            remove_columns=["question", "response_A", "response_B", "label", "injected_bias", "domain"],
        )

    else:
        dataset = load_dataset(
            "CohereLabsCommunity/multilingual-reward-bench",
            data_name,
            split="test",
            cache_dir=cache_dir,
        )

    if "category" not in dataset.features:
        dataset = dataset.map(lambda _: {"category": "all"})

    return dataset


def load_listwise_dataset(
    dataset_name: str = "allenai/reward-bench-2",
    cache_dir: str = default_cache_dir,
    split: str = "test",
) -> Dataset:
    """Load a listwise (1-of-4) evaluation dataset."""
    dataset = load_dataset(dataset_name, cache_dir=cache_dir, split=split)

    def _flatten(example):
        example["chosen"] = example["chosen"][0]
        for i in range(3):
            example[f"rejected_{i}"] = example["rejected"][i]
        return example

    dataset = dataset.map(_flatten)

    if "category" not in dataset.features:
        dataset = dataset.map(lambda _: {"category": "all"})

    return dataset

# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

class _SafeFormatDict(dict):
    """Keep unknown template placeholders untouched during formatting."""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"

def build_pairwise_prompts(example: dict, prompt_template: str) -> dict:
    """Shuffle chosen/rejected into A/B and build the user prompt."""
    answer_a = example["chosen"]
    answer_b = example["rejected"]

    is_shuffled = np.random.rand() < 0.5
    if is_shuffled:
        answer_a, answer_b = answer_b, answer_a
        winner_text = "B"
    else:
        winner_text = "A"

    format_values = _SafeFormatDict({
        "question": example["prompt"],
        "answer_a": answer_a,
        "answer_b": answer_b,
    })
    # Include any extra fields from the example (e.g. "rubric") for templates
    # that reference them.
    for key in example:
        if key not in format_values:
            format_values[key] = example[key]

    user_content = prompt_template.format_map(format_values)

    return {
        "user_prompt": user_content,
        "winner": winner_text,
        "answer_a": answer_a,
        "answer_b": answer_b,
        "category": example["category"],
    }


def build_listwise_prompts(example: dict, prompt_template: str) -> dict:
    """Shuffle the chosen answer into a random position among 4 options."""
    answer_a = example["chosen"]
    answer_b = example["rejected_0"]
    answer_c = example["rejected_1"]
    answer_d = example["rejected_2"]

    shuffle_option = np.random.randint(0, 4)
    winner_text = "A"

    if shuffle_option == 1:
        answer_a, answer_b = answer_b, answer_a
        winner_text = "B"
    elif shuffle_option == 2:
        answer_a, answer_c = answer_c, answer_a
        winner_text = "C"
    elif shuffle_option == 3:
        answer_a, answer_d = answer_d, answer_a
        winner_text = "D"

    format_values = _SafeFormatDict({
        "question": example["prompt"],
        "answer_a": answer_a,
        "answer_b": answer_b,
        "answer_c": answer_c,
        "answer_d": answer_d,
    })
    for key in example:
        if key not in format_values:
            format_values[key] = example[key]

    user_content = prompt_template.format_map(format_values)

    return {
        "prompt": example["prompt"],
        "user_prompt": user_content,
        "winner": winner_text,
        "answer_a": answer_a,
        "answer_b": answer_b,
        "answer_c": answer_c,
        "answer_d": answer_d,
        "category": example.get("category", "all"),
    }


def split_by_category(dataset: Dataset) -> DatasetDict:
    """Split a dataset into per-category subsets."""
    categories = dataset.unique("category")
    subsets = {}
    for cat in categories:
        subsets[cat] = dataset.filter(lambda x: x["category"] == cat).remove_columns(["category"])
    return DatasetDict(subsets)
