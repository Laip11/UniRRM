# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
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

import asyncio
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import List

import aiohttp
import requests
import torch
from verl.workers.reward_manager import register
from typing import Union,Optional
from verl import DataProto


@dataclass
class CallScoreFuncInput:
    data_source: Optional[str]
    solution_str: str
    ground_truth: str
    extra_info: Optional[dict]
    prompt_str: Optional[str] = None
    response_str: Optional[str] = None
    valid_response_length: Optional[int] = None


@dataclass
class ScoreFuncOutput:
    score: Union[float,int]
    detail: dict

async def async_call_online_reward_model(url: str, **kwargs):
    headers = {"accept": "application/json",
                "Content-Type": "application/json"}

    json_data = {**kwargs}

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=json_data) as response:
                response_text = await response.text()

                if response.status != 200:
                    raise aiohttp.ClientResponseError(
                        response.request_info,
                        response.history,
                        status=response.status,
                        message=f"HTTP {response.status} from {url}. Response body: {response_text}"
                    )

                try:
                    res = await response.json()
                except Exception as json_err:
                    raise ValueError(f"Failed to parse JSON response from {url}. Raw response: {response_text}") from json_err

                if "score" not in res:
                    raise ValueError(f"Reward server response missing 'score'. URL: {url}, Response: {res}")

                final_score = res["score"]

                return float(final_score), res
    except aiohttp.ClientError as e:
        raise ConnectionError(f"Network error calling reward API {url}: {type(e).__name__}: {str(e)}. Request data: {json_data}") from e
    except asyncio.TimeoutError as e:
        raise TimeoutError(f"Timeout calling reward API {url}. Request data: {json_data}") from e


def call_online_reward_model(url: str, **kwargs):
    # 同步调用 async 函数
    loop = asyncio.get_event_loop()

    score, details = loop.run_until_complete( async_call_online_reward_model(url, **kwargs) )

    return score, details
# async def async_call_online_reward_model(url: str, **kwargs):
#     try:
#         headers = {
#             "accept": "application/json",
#             "Content-Type": "application/json",
#         }

#         json_data = {**kwargs}

#         async with aiohttp.ClientSession() as session:
#             async with session.post(url, headers=headers, json=json_data) as response:
#                 res = await response.json()

#                 final_score = res.get("score")
#                 return float(final_score), res
#     except Exception as e:
#         return 0.0, {}


# def call_online_reward_model(url: str, **kwargs):
#     try:
#         # Use the async function in a synchronous context
#         loop = asyncio.get_event_loop()
#         score, details = loop.run_until_complete(
#             async_call_online_reward_model(url, **kwargs)
#         )
#         return score, details
#     except Exception as e:
#         return 0.0, {}


_default_compute_score = call_online_reward_model


@register("naive")
class NaiveRewardManager:
    """The reward manager."""

    def __init__(
        self,
        reward_api,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key="data_source",
    ) -> None:
        self.reward_api = reward_api
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        self.reward_fn_key = reward_fn_key

    async def async_compute_score(self, input_item: CallScoreFuncInput) -> ScoreFuncOutput:
        """异步处理单个评分请求"""
        try:
            score, detail = await async_call_online_reward_model(
                url=self.reward_api,
                data_source=input_item.data_source,
                solution_str=input_item.solution_str,
                ground_truth=input_item.ground_truth,
                extra_info=input_item.extra_info,
                prompt_str=input_item.prompt_str,
                response_str=input_item.response_str,
            )
            return ScoreFuncOutput(score=score, detail=detail)
        except Exception as e:
            error_context = (
                f"Error computing score for data_source='{input_item.data_source}'. "
                f"prompt_str (first 200 chars): '{input_item.prompt_str[:200] if input_item.prompt_str else None}...', "
                f"response_str (first 200 chars): '{input_item.response_str[:200] if input_item.response_str else None}...'. "
                f"Original error: {type(e).__name__}: {str(e)}"
            )
            raise Exception(error_context) from e


    async def batch_compute_scores(self, input_list: List[CallScoreFuncInput], max_concurrency: int = 40) -> List[ScoreFuncOutput]:
        """并行处理所有评分请求，限制最大并发数量"""
        # 创建信号量来控制并发数
        semaphore = asyncio.Semaphore(max_concurrency)

        async def bounded_compute_score(input_item: CallScoreFuncInput):
            async with semaphore:  # 使用信号量限制并发
                return await self.async_compute_score(input_item)

        # 为每个输入创建任务，但并发会被信号量限制
        tasks = [bounded_compute_score(input_item) for input_item in input_list]

        # gather确保结果顺序与输入顺序一致
        return await asyncio.gather(*tasks)

    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        # reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        scorefuncinput_list: List[CallScoreFuncInput] = []

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][
                :prompt_length
            ].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][
                prompt_length:
            ].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]

            data_source = data_item.non_tensor_batch["data_source"]

            extra_info = data_item.non_tensor_batch.get("extra_info", None)
            prompt_str = self.tokenizer.decode(valid_prompt_ids)
            response_str = self.tokenizer.decode(valid_response_ids)

            scorefuncinput_list.append(
                CallScoreFuncInput(
                    data_source=data_source,
                    solution_str=sequences_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                    prompt_str=prompt_str,
                    response_str=response_str,
                    valid_response_length=valid_response_length,
                )
            )

        # 并行处理所有评分请求
        score_funcoutput_list = asyncio.run(
            self.batch_compute_scores(scorefuncinput_list)
        )

        # 处理结果
        for i in range(len(data)):
            data_source = scorefuncinput_list[i].data_source
            valid_response_length = scorefuncinput_list[i].valid_response_length
            sequences_str = scorefuncinput_list[i].solution_str
            score = score_funcoutput_list[i].score
            reward_tensor[i, valid_response_length - 1] = score

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(sequences_str)

        score_detail_list = [output.detail for output in score_funcoutput_list]

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_detail": score_detail_list,  # 自己定义的detail详情
                # "reward_extra_info": reward_extra_info,  # verl定义的detail info
            }
        else:
            return reward_tensor, score_detail_list