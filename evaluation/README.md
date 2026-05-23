# UniRRM Evaluation

This directory contains the evaluation pipeline for UniRRM and other reward models. It supports pairwise, listwise, and pointwise-on-pairwise evaluation with vLLM-based inference.

## Environment

Use the `verl` environment from the project root README. The evaluation and inference pipeline is based on this environment because it only depends on vLLM and the standard model/tokenizer stack.

```bash
conda activate verl
cd evaluation
```

## Quick Start

Run pairwise evaluation on the default benchmark:

```bash
bash script/run_eval_pair_wise.sh
```

Run listwise evaluation on RewardBench v2:

```bash
bash script/run_eval_list_wise.sh
```

Run pointwise scoring on pairwise benchmarks:

```bash
bash script/run_eval_pointwise_on_pair_benchmark.sh
```

By default, these scripts evaluate `SUSTech-NLP/UniRRM-8B` with the `UniRRM` prompt template.

## Evaluation Modes

### Pairwise Evaluation

`evaluation_pairwise.py` evaluates two candidate responses for each prompt. It uses:

- `load_pairwise_dataset` to load and normalize benchmark data.
- `build_pairwise_prompts` to format two responses into the model prompt.
- `compute_pairwise_accuracy` to report accuracy and parse failures by category.

The default launch script is:

```bash
bash script/run_eval_pair_wise.sh
```

### Listwise Evaluation

`evaluation_listwise.py` evaluates four candidate responses and asks the model to select the best one. It uses:

- `load_listwise_dataset` to load RewardBench v2-style data.
- `build_listwise_prompts` to format four candidate responses.
- `compute_listwise_accuracy` to compute overall accuracy and parse failures.

The default launch script is:

```bash
bash script/run_eval_list_wise.sh
```

### Pointwise-on-Pairwise Evaluation

`evaluation_pointwise_on_pair_benchmark.py` scores each response independently on pairwise benchmarks. The response with the higher score is selected as the winner.

This mode is useful for pointwise scoring reward models. It uses `genrm-scorer` or `scalar-rm` as the reward type.

The default launch script is:

```bash
bash script/run_eval_pointwise_on_pair_benchmark.sh
```

## Common Options

All evaluation entry points share the same core arguments:

- `--model_name_or_path`: Hugging Face model ID or local model path.
- `--datasets`: One or more dataset names.
- `--reward_type`: Inference mode. Supported values include `genrm-judge`, `genrm-scorer`, and `scalar-rm`, depending on the entry point.
- `--template_name`: Prompt template key registered in `src/templates/__init__.py`.
- `--temperature`, `--top_p`, `--top_k`, `--max_new_tokens`: Generation parameters.
- `--debug`: Run on a small subset for quick checks.

Example direct command:

```bash
python3 evaluation_pairwise.py \
  --model_name_or_path SUSTech-NLP/UniRRM-8B \
  --datasets judgebench_pro \
  --reward_type genrm-judge \
  --template_name UniRRM \
  --temperature 0
```

You can also override script defaults with environment variables:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
DATASETS="judgebench_pro MM-Eval" \
TEMPLATE_NAME=UniRRM \
REWARD_TYPE=genrm-judge \
bash script/run_eval_pair_wise.sh
```

## Reward Types

`genrm-judge` sends all candidate responses in one prompt and asks the model to directly output the winner.

`genrm-scorer` scores each candidate response independently with a generative model, then selects the response with the highest parsed score.

`scalar-rm` uses a scalar reward model through vLLM's pooling runner and selects the response with the highest scalar reward.

## Templates

Prompt templates live in `src/templates/`. Each model template is an `EvalPromptTemplate` that can define:

- `system_template_list` and `user_template_list` for pairwise/listwise judging.
- `system_template_point` and `user_template_point` for pointwise scoring.
- `enable_thinking` for models that support thinking mode.
- `listwise_answer_extractor` and `pointwise_answer_extractor` for parsing model outputs.

To add a new model:

1. Create a new file under `src/templates/`, following examples such as `unirrm.py`.
2. Define an `EvalPromptTemplate` with the model's prompt format and output parser.
3. Import and register it in `TEMPLATE_REGISTRY` in `src/templates/__init__.py`.
4. Run evaluation with `--template_name YOUR_TEMPLATE_NAME`.

## Datasets

Dataset loading and field conversion are implemented in `src/data_loader.py`.

For pairwise evaluation, `load_pairwise_dataset` should return examples with:

- `prompt`: user input.
- `chosen`: preferred response.
- `rejected`: non-preferred response.
- `category`: evaluation category. If missing, it is filled with `all`.

For listwise evaluation, `load_listwise_dataset` should return examples that can be converted into:

- `prompt`: user input.
- `chosen`: best response.
- `rejected_0`, `rejected_1`, `rejected_2`: other candidate responses.
- `category`: evaluation category. If missing, it is filled with `all`.

To evaluate a new dataset:

1. Add a new branch in `load_pairwise_dataset` or `load_listwise_dataset`.
2. Load the benchmark with `datasets.load_dataset` or a local data source.
3. Convert raw fields into the expected schema above.
4. Pass the dataset name through `--datasets` or the `DATASETS` environment variable.

## Outputs

Evaluation results are appended to:

```text
res/<reward_type>/<model_name>.txt
```

If a dataset name already appears in the result file, the script skips that dataset to avoid duplicate logging. Remove or rename the result file when you want to rerun the same dataset.

## Debugging

Use `--debug` for a quick smoke test before launching a full benchmark:

```bash
python3 evaluation_pairwise.py \
  --model_name_or_path SUSTech-NLP/UniRRM-8B \
  --datasets judgebench_pro \
  --reward_type genrm-judge \
  --template_name UniRRM \
  --debug
```

If parse failures are high, check that the selected template's answer extractor matches the model's output format.
