import argparse
from typing import Any, Dict, List, Optional

from tqdm import trange
from vllm import LLM, SamplingParams


def generate_with_vllm(
    llm: LLM,
    prompts: List[str],
    tokenizer,
    need_apply_chat_template: bool = True,
    system_prompt: Optional[str] = None,
    enable_thinking: bool = False,
    max_tokens: int = 4096,
    temperature: float = 0.0,
    top_k: int = 1,
    top_p: float = 0.9,
) -> List[Dict[str, Any]]:
    """Generate responses for a batch of prompts using a vLLM engine.

    Parameters
    ----------
    llm : LLM
        A loaded vLLM ``LLM`` instance.
    prompts : list[str]
        User prompts to generate responses for.
    tokenizer
        The tokenizer associated with the model.
    need_apply_chat_template : bool
        If True, wrap each prompt using the tokenizer's chat template.
    system_prompt : str | None
        Optional system message prepended to the conversation.
    enable_thinking : bool
        If True, pass ``enable_thinking=True`` to the chat template (for
        models that support a chain-of-thought thinking mode).
    max_tokens : int
        Maximum number of new tokens to generate.
    temperature : float
        Sampling temperature.
    top_k : int
        Top-k sampling parameter.
    top_p : float
        Nucleus sampling probability.

    Returns
    -------
    list[dict]
        A list of dicts, each containing at least a ``"response"`` key with
        the generated text.
    """
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )

    if need_apply_chat_template:
        formatted_prompts = []
        for prompt in prompts:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            chat_kwargs = {"tokenize": False, "add_generation_prompt": True}
            if enable_thinking:
                chat_kwargs["enable_thinking"] = True

            formatted = tokenizer.apply_chat_template(messages, **chat_kwargs)
            formatted_prompts.append(formatted)
    else:
        formatted_prompts = list(prompts)

    outputs = llm.generate(formatted_prompts, sampling_params)

    results = []
    for output in outputs:
        generated_text = output.outputs[0].text
        results.append({"response": generated_text})

    return results


def create_common_parser(
    description: str,
    reward_type_choices: List[str],
) -> argparse.ArgumentParser:
    """Create an argument parser with common evaluation arguments.

    Each entry-point script can extend the returned parser with additional
    arguments before calling ``parse_args()``.

    Parameters
    ----------
    description : str
        CLI description shown in ``--help``.
    reward_type_choices : list[str]
        Valid values for ``--reward_type``.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--datasets", type=str, nargs="+", required=True,
                        help="List of dataset names to evaluate")
    parser.add_argument("--model_name_or_path", type=str, required=True,
                        help="Path to the local model")
    parser.add_argument("--temperature", type=float, default=0,
                        help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Nucleus sampling probability")
    parser.add_argument("--top_k", type=int, default=1,
                        help="Top-k sampling parameter")
    parser.add_argument("--max_new_tokens", type=int, default=4096,
                        help="Maximum output tokens")
    parser.add_argument("--reward_type", type=str, choices=reward_type_choices,
                        required=True, help="Type of reward model")
    parser.add_argument("--template_name", type=str, required=True,
                        help="Prompt template name")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode (use fewer samples)")
    return parser


def print_args(args):
    """Pretty-print all parsed CLI arguments."""
    print("\n" + "=" * 50)
    print("[ Arguments ]")
    max_key_len = max(len(k) for k in vars(args))
    for key, value in vars(args).items():
        print(f"{key:<{max_key_len + 2}}: {value}")
    print("=" * 50 + "\n")