import re
import string
import math
from openai import OpenAI
from typing import Union 
import time
import random
from openai import OpenAI
from openai import APIError, RateLimitError, APIConnectionError, Timeout

class OpenAIClient:
    def __init__(self, 
                 api_key=None, 
                 base_url="https://api.openai.com/v1",
                 model="gpt-4o-mini",
                 seed=42,
                 max_retries=3):

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.seed = seed
        self.max_retries = max_retries

    def get_response_chat(self, query, max_tokens=10, temperature=0):
        messages = [{"role": "user", "content": query}]

        for attempt in range(self.max_retries):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    seed=self.seed
                )
                return completion.choices[0].message.content

            # ----- Error Cases -----

            except RateLimitError as e:
                # 调用次数太多（最常见）
                wait = 2 ** attempt + random.random()
                print(f"[RateLimit] retrying in {wait:.2f}s ...")
                time.sleep(wait)

            except APIConnectionError as e:
                # 网络不稳定 / DNS / 连接失败
                wait = 2 ** attempt + random.random()
                print(f"[ConnectionError] retrying in {wait:.2f}s ...")
                time.sleep(wait)

            except Timeout as e:
                # 请求超时
                wait = 2 ** attempt + random.random()
                print(f"[Timeout] retrying in {wait:.2f}s ...")
                time.sleep(wait)

            except APIError as e:
                # 服务器内部错误（可重试）
                if e.status >= 500:
                    wait = 2 ** attempt + random.random()
                    print(f"[ServerError {e.status}] retrying in {wait:.2f}s ...")
                    time.sleep(wait)
                else:
                    # 4xx 类型一般不可恢复（如 invalid request）
                    print(f"[APIError] non-retryable: {e}")
                    raise e

        raise Exception("Failed after max retry attempts.")


def pairwise_accuracy(prediction: str, golden_answer: str) -> int:
    return 1 if prediction.strip() == golden_answer.strip() else 0

def pointwise_rewawrd_score(prediction: int, golden_answer: int) -> float:
    return abs(prediction - golden_answer)


def listwise_reward_score_1(prediction: str, golden_answer: str) -> int:
    '''
    The first approach to compute listwise reward score.
    if  the model can determine the best sample among the predictions.
    if yes, return 1.0 else return 0.0
    '''
    return 1 if prediction.strip() == golden_answer.strip() else 0


def listwise_reward_score_2(prediction: list[str], golden_answers: list[str]) -> float:
    # The second approach to compute listwise reward score.
    # Calculate the nDCG score

    # transform to lower case and strip spaces
    prediction = [p.lower().strip() for p in prediction]
    golden_answers = [g.lower().strip() for g in golden_answers]
    # get the length of golden answers
    if set(prediction) != set(golden_answers):
        return 0
    else:
        k = len(golden_answers)
        relevance_map = {item: k - idx for idx, item in enumerate(golden_answers)}  # higher score for earlier items

        dcg = 0.0
        for i, item in enumerate(prediction[:k]):
            rel = relevance_map.get(item, 0)
            if rel:
                dcg += rel / math.log2(i + 2)

        # build ideal DCG by sorting relevance scores descending
        ideal_rels = sorted(relevance_map.values(), reverse=True)[:k]
        idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal_rels))

        return 0.0 if idcg == 0 else dcg / idcg

def format_reward(predict_str: str) -> float:
    pattern = re.compile(r"<evaluation>(.*?)</evaluation>.*?<answer>(.*?)</answer>", re.DOTALL)
    match_result = re.fullmatch(pattern, predict_str)
    return 1.0 if match_result else 0.0

def rubric_score(rubric_str: str) -> float:
    rubric_judge_prompt = '''

'''

    judge_model = OpenAIClient(base_url="https://api.openai.com/v1", 
                               api_key=None,
                               model="gpt-4o-mini")
    score = judge_model.get_response_chat(query=rubric_str, max_tokens=10, temperature=0)

    return float(score.strip())

def test_judge(rubric_str: Union[str,int,float], ground_truth: Union[str,int,float]) -> float:
    judge_prompt = '''
Please determine whether the following two answers are equivalent.

[Answer 1]
{rubric_str}

[Answer 2]
{ground_truth}

If they are equivalent, then directly return 1; if they are not equivalent, directly return 0. Please do not output anything else.

'''
    input = judge_prompt.format(rubric_str=rubric_str, ground_truth=ground_truth)
    judge_model = OpenAIClient(base_url="https://api.openai.com/v1", 
                               api_key=None,
                               model="gpt-4o-mini")
    score = judge_model.get_response_chat(query=input, max_tokens=10, temperature=0)
    return float(score.strip())


def compute_score(predict_str: str, ground_truth: dict,weight1: float, weight2: float, weight3: float) -> float:
    # pairwise_acc = pairwise_accuracy(predict_str, ground_truth['pairwise'])
    # pointwise_score = pointwise_rewawrd_score(int(predict_str.strip()), int(ground_truth['pointwise']))
    # listwise_score = listwise_reward_score_1(predict_str, ground_truth['listwise'])
    # format_score = format_reward(predict_str)

    # total_score = (weight1 * pairwise_acc) + (weight2 * (1 - pointwise_score)) + (weight3 * listwise_score) + (0.1 * format_score)
    test_score = test_judge(predict_str, ground_truth)
    return test_score