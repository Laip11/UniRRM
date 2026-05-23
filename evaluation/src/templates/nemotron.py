"""Llama-3.3-Nemotron-Super-49B-GenRM-Multilingual template."""

from ._base import EvalPromptTemplate, make_extractor, make_pointwise_extractor

Llama_3_3_Nemotron_Super_49B_GenRM_Multilingual = EvalPromptTemplate(
    template_name="Llama_3_3_Nemotron_Super_49B_GenRM_Multilingual",
    listwise_answer_extractor=make_extractor(
        r"\\boxed\{\s*(\d+)\s*,\s*(\d+)\s*\}", answer_type=int
    ),
    pointwise_answer_extractor=make_pointwise_extractor(
        r"\\boxed\{\s*(\d+)"
    ),
    system_template_list=None,
    user_template_list=None,
    formatted_prompt="""
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

detailed thinking on
<|eot_id|><|start_header_id|>user<|end_header_id|>

You are a skilled little expert at scoring responses. You should evaluate given responses based on the given judging criteria.
Given the context of the conversation (the last turn is the User's query) and one or two responses from the Assistant, you need to refer to the [Helpfulness Scoring Guidelines] to score each individual response.
If there are two responses, you need to also give a ranking score based on the [Ranking Scoring Guidelines].
Before scoring, please analyze step by step. Your scoring needs to be as strict as possible.

[Helpfulness Scoring Guidelines]

When evaluating Helpfulness, consider the following factors:

- Correctness/Completeness: Is the response accurate and complete?
- Coherence/Clarity: Is the response clear, coherent, and easy to understand?
- Instruction following: Does the response follow the instructions and fulfill the user's request?
- Relevance: Is the response relevant to the user's query/input?
- Level of Detail and Creativity: Does the response provide enough detail without being too verbose? Does it show creativity but not hallucinations?

**Score 5: Extremely Helpful**

- The response is extremely helpful and completely aligned with the spirit of what the prompt was asking for.
- It accurately acts on the user's request, without unnecessary information.
- If a user request is not possible/in line with desired model behavior, a helpful response provides useful context and rationale.

**Score 4: Mostly Helpful**

- The response is mostly helpful and mainly aligned with what the user was looking for.
- There is still some room for improvement, but the response is generally useful.

**Score 3: Partially Helpful**

- The response is partially helpful but misses the overall goal of the user's query/input in some way.
- The response did not fully satisfy what the user was looking for.

**Score 2: Borderline Unhelpful**

- The response is borderline unhelpful and mostly does not capture what the user was looking for.
- However, it is still usable and helpful in a small way.

**Score 1: Not Helpful**

- The response is not useful or helpful at all.
- The response completely missed the essence of what the user wanted.

[Ranking Scoring Guidelines]

Ranking score is used to rank the two responses based on their helpfulness. Even if you give the same individual helpfulness score for both responses, you need to differentiate them strictly.
The ranking score is a number between 1 and 6, where:
1 = Response 1 is much better than Response 2
2 = Response 1 is better than Response 2
3 = Response 1 is slightly better than Response 2
4 = Response 2 is slightly better than Response 1
5 = Response 2 is better than Response 1
6 = Response 2 is much better than Response 1

#### Conversation Context ####
User: {question}

#### Responses to be Scored ####
[The Begin of Response 1]
{answer_a}
[The End of Response 1]
[The Begin of Response 2]
{answer_b}
[The End of Response 2]
#### Output Format Requirements ####
First give your analysis on each responses in the format of:
[The Begin of Analysis on Response i]
Analysis on the i-th response
[The End of Analysis on Response i]

Then give the scores of each response in order, separate by comma in the boxed, adhering this format:
[The Begin of Individual Scores]
\\boxed{{x, y}} if there exists 2 responses
[The End of Individual Scores]

If there are two responses, give the relative ranking score in the format of:
[The Begin of Ranking Score]
\\boxed{{z}} 
[The End of Ranking Score]
You don't need to give a ranking score if only one response is provided.
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
""",
)
