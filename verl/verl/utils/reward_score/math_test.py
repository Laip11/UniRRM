# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Adapted from https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/hendrycks_math/utils.py
from openai import OpenAI
from typing import Union
from openai import APIError, RateLimitError, APIConnectionError, Timeout
import time
import random
import requests


def compute_score1(data_source, solution_str, ground_truth, extra_info=None, **kwargs) -> float:
    """
    调用 Reward Server 计算 score
    """
    retval = 0.0

    REWARD_SERVER_URL = "http://localhost:9999/reward"  # 你的 FastAPI reward server 地址


    try:
        # 如果 solution_str 中有 boxed 内容，可以先提取
        string_in_last_boxed = last_boxed_only_string(solution_str)
        if string_in_last_boxed is not None:
            answer = remove_boxed(string_in_last_boxed)
        else:
            answer = solution_str

        # 构造请求 JSON
        payload = {
            "data_source": data_source,
            "solution_str": answer,
            "ground_truth": ground_truth,
            "extra_info": extra_info or {}
        }

        # 调用 reward server
        r = requests.post(REWARD_SERVER_URL, json=payload, timeout=300)
        r.raise_for_status()  # 请求失败就会抛异常

        score_dict = r.json()
        retval = float(score_dict.get("score", 0.0))  # 默认 0.0

    except Exception as e:
        print("compute_score error:", e)

    return retval

# class OpenAIClient:
#     def __init__(self, 
#                  api_key='a123', 
#                  base_url="http://127.0.0.1:7007/v1",
#                  model="Qwen2.5-72B-Instruct",
#                  seed=42,
#                  max_retries=3):

#         self.client = OpenAI(api_key=api_key, base_url=base_url)
#         self.model = model
#         self.seed = seed
#         self.max_retries = max_retries

#     def get_response_chat(self, query, max_tokens=10, temperature=0):
#         messages = [{"role": "user", "content": query}]

#         for attempt in range(self.max_retries):
#             try:
#                 completion = self.client.chat.completions.create(
#                     model=self.model,
#                     messages=messages,
#                     max_tokens=max_tokens,
#                     temperature=temperature,
#                     seed=self.seed
#                 )
#                 return completion.choices[0].message.content

#             # ----- Error Cases -----

#             except RateLimitError as e:
#                 # 调用次数太多（最常见）
#                 wait = 2 ** attempt + random.random()
#                 print(f"[RateLimit] retrying in {wait:.2f}s ...")
#                 time.sleep(wait)

#             except APIConnectionError as e:
#                 # 网络不稳定 / DNS / 连接失败
#                 wait = 2 ** attempt + random.random()
#                 print(f"[ConnectionError] retrying in {wait:.2f}s ...")
#                 time.sleep(wait)

#             except Timeout as e:
#                 # 请求超时
#                 wait = 2 ** attempt + random.random()
#                 print(f"[Timeout] retrying in {wait:.2f}s ...")
#                 time.sleep(wait)

#             except APIError as e:
#                 # 服务器内部错误（可重试）
#                 if e.status >= 500:
#                     wait = 2 ** attempt + random.random()
#                     print(f"[ServerError {e.status}] retrying in {wait:.2f}s ...")
#                     time.sleep(wait)
#                 else:
#                     # 4xx 类型一般不可恢复（如 invalid request）
#                     print(f"[APIError] non-retryable: {e}")
#                     raise e

#         raise Exception("Failed after max retry attempts.")
    
    
# def test_judge(rubric_str: Union[str,int,float], ground_truth: Union[str,int,float]) -> float:
#     judge_prompt = '''
# Please determine whether the following two answers are equivalent.

# [Answer 1]
# {rubric_str}

# [Answer 2]
# {ground_truth}

# If they are equivalent, then directly return 1; if they are not equivalent, directly return 0. Please do not output anything else.

# '''
#     input = judge_prompt.format(rubric_str=rubric_str, ground_truth=ground_truth)
#     judge_model = OpenAIClient(api_key='a123', 
#                                 base_url="http://127.0.0.1:7007/v1",
#                                 model="Qwen2.5-72B-Instruct",)
#     score = judge_model.get_response_chat(query=input, max_tokens=10, temperature=0)
#     return float(score.strip())

# def compute_score(data_source,solution_str, ground_truth,extra_info,**kwargs) -> float:
#     retval = 0.0
#     try:
#         string_in_last_boxed = last_boxed_only_string(solution_str)
#         if string_in_last_boxed is not None:
#             answer = remove_boxed(string_in_last_boxed)
#             if test_judge(answer, ground_truth):
#             #if is_equiv(answer, ground_truth):
#                 retval = 1.0
#     except Exception as e:
#         print(e)

#     return retval


# string normalization from https://github.com/EleutherAI/lm-evaluation-harness/blob/master/lm_eval/tasks/hendrycks_math.py
def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except Exception:
        return str1 == str2


def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"

    assert s[: len(left)] == left
    assert s[-1] == "}"

    return s[len(left) : -1]


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    retval = None if right_brace_idx is None else string[idx : right_brace_idx + 1]

    return retval


def fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:  # noqa: E722
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:  # noqa: E722
        return string


def remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string


def fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def strip_string(string):
    # linebreaks
    string = string.replace("\n", "")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")  # noqa: W605

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1).
    # Also does a/b --> \\frac{a}{b}
    string = fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = fix_a_slash_b(string)

    return string
