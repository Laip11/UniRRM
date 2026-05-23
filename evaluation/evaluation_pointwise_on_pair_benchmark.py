"""Pointwise-on-pairwise evaluation entry point.

Evaluates a pointwise scoring model on pairwise benchmarks: each candidate
response is scored independently, and the higher-scoring response wins.

This is functionally equivalent to running evaluation_pairwise.py with
reward_type=genrm-scorer, but kept as a separate entry point for clarity.

Pipeline: data loading -> pointwise inference -> result extraction & accuracy.
"""

import os

from tqdm import tqdm

from src.templates import TEMPLATE_REGISTRY
from src.utils import print_args, create_common_parser
from src.data_loader import (
    load_pairwise_dataset,
    build_pairwise_prompts,
    split_by_category,
)
from src.inference import create_engine, PAIRWISE_ANSWER_KEYS
from src.result_calculator import (
    compute_pairwise_accuracy,
    save_pairwise_report,
)

REWARD_TYPE_CHOICES = [
    "genrm-scorer",
    "scalar-rm",
]


def get_args():
    parser = create_common_parser("Pointwise evaluation on pairwise benchmarks", REWARD_TYPE_CHOICES)
    return parser.parse_args()

def main():
    args = get_args()

    # --- Template ---
    assert args.template_name in TEMPLATE_REGISTRY, f"Unknown template: {args.template_name}"
    model_template = TEMPLATE_REGISTRY[args.template_name]

    pairwise_prompt = (
        model_template.user_template_list
        if model_template.user_template_list is not None
        else model_template.formatted_prompt
    )
    need_apply_chat_template = model_template.user_template_list is not None

    # --- Logging ---
    model_name = args.model_name_or_path.split("/")[-1]
    log_file = f"res/{args.reward_type}/{model_name}.txt"
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    print_args(args)

    # --- Inference engine (created once, reused across datasets) ---
    engine = create_engine(
        reward_type=args.reward_type,
        model_name_or_path=args.model_name_or_path,
        answer_keys=PAIRWISE_ANSWER_KEYS,
    )

    # --- Process each dataset ---
    for data_name in tqdm(args.datasets, position=0, leave=True, desc="Processing datasets"):

        # Skip if already evaluated
        with open(log_file, "a+") as fh:
            fh.seek(0)
            if data_name in fh.read():
                print(f"Results for {data_name} already logged in {log_file}. Skipping.")
                continue

        # --- 1. Data loading ---
        eval_data = load_pairwise_dataset(data_name)
        if args.debug:
            eval_data = eval_data.select(range(5))

        eval_data = eval_data.map(
            lambda x: build_pairwise_prompts(x, pairwise_prompt),
            remove_columns=[],
        )
        category_datasets = split_by_category(eval_data)

        # --- 2. Inference (per category) ---
        category_results = {}
        for category, dataset in tqdm(category_datasets.items(), desc="Running inference"):
            print(f"Running pointwise inference for category: {category}")
            results = engine.infer(dataset, model_template, need_apply_chat_template, args)
            category_results[category] = results

        # --- 3. Result extraction & accuracy ---
        category_accuracy, missed_count = compute_pairwise_accuracy(
            category_datasets, category_results, model_template,
        )
        save_pairwise_report(log_file, data_name, category_accuracy, missed_count)


if __name__ == "__main__":
    main()
