"""Listwise evaluation entry point.

Pipeline: data loading -> inference -> result extraction & accuracy.
"""

import os

from tqdm import tqdm

from src.templates import TEMPLATE_REGISTRY
from src.utils import print_args, create_common_parser
from src.data_loader import load_listwise_dataset, build_listwise_prompts
from src.inference import create_engine, LISTWISE_ANSWER_KEYS
from src.result_calculator import compute_listwise_accuracy, save_listwise_report

REWARD_TYPE_CHOICES = [
    "genrm-judge",
    "genrm-scorer",
    "scalar-rm",
]

DATASETS = {
    "reward-bench-2": "allenai/reward-bench-2",
}

DEFAULT_LISTWISE_USER_PROMPT = """
<User_Input>
{question}
</User_Input>

<Response1>
{answer_a}
</Response1>

<Response2>
{answer_b}
</Response2>

<Response3>
{answer_c}
</Response3>

<Response4>
{answer_d}
</Response4>
"""


def get_args():
    parser = create_common_parser(
        "Listwise evaluation",
        REWARD_TYPE_CHOICES    )
    return parser.parse_args()


def main():
    args = get_args()

    # --- Template ---
    assert args.template_name in TEMPLATE_REGISTRY, f"Unknown template: {args.template_name}"
    model_template = TEMPLATE_REGISTRY[args.template_name]

    # Use the template's user_template_list if available, otherwise fall back
    # to the default 4-response prompt and use formatted_prompt as last resort.
    if model_template.user_template_list is not None:
        listwise_prompt = model_template.user_template_list
        need_apply_chat_template = True
    elif model_template.formatted_prompt is not None:
        listwise_prompt = model_template.formatted_prompt
        need_apply_chat_template = False
    else:
        listwise_prompt = DEFAULT_LISTWISE_USER_PROMPT
        need_apply_chat_template = True

    # --- Logging ---
    model_name = args.model_name_or_path.split("/")[-1]
    log_file = f"res/{args.reward_type}/{model_name}.txt"
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    print_args(args)

    # --- Inference engine (created once, reused across datasets) ---
    engine = create_engine(
        reward_type=args.reward_type,
        model_name_or_path=args.model_name_or_path,
        answer_keys=LISTWISE_ANSWER_KEYS,
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
        eval_data = load_listwise_dataset(data_name, split="test")
        eval_data = eval_data.map(
            lambda x: build_listwise_prompts(x, listwise_prompt),
            remove_columns=[],
        )
        if args.debug:
            eval_data = eval_data.select(range(10))

        # --- 2. Inference ---
        results = engine.infer(eval_data, model_template, need_apply_chat_template, args)

        # --- 3. Result extraction & accuracy ---
        accuracy, invalid_count, total_len = compute_listwise_accuracy(
            eval_data, results, model_template,
        )

        if total_len == 0:
            print(f"No predictions to evaluate for {data_name}!")
        else:
            print(f"[{data_name}] Accuracy: {accuracy:.4f}  (total={total_len}, parse_failures={invalid_count})")
            save_listwise_report(log_file, data_name, accuracy, invalid_count, total_len)


if __name__ == "__main__":
    main()
