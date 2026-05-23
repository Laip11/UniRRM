"""JudgeLRM template."""

from ._base import EvalPromptTemplate, make_extractor, make_pointwise_extractor

JudgeLRM = EvalPromptTemplate(
    template_name="JudgeLRM",
    enable_thinking=True,
    listwise_answer_extractor=make_extractor(r"<answer>(\d+)</answer>", answer_type=int),
    pointwise_answer_extractor=make_pointwise_extractor(r"<answer>(\d+)</answer>"),
    system_template_list="""
    You are a helpful assistant. The assistant first performs a detailed,
step-by-step reasoning process in its mind and then provides the user with
the answer. The reasoning process and answer are enclosed within  and <answer> </answer> tags, respectively, i.e., <answer> answer here </answer>. Now the user asks you
to judge the performance of two AI assistants in response to the question.
Score assistants 1-10 (higher=better). Criteria includes helpfulness,
relevance, accuracy, and level of detail. Avoid order, length, style or
other bias. After thinking, when you finally reach a conclusion, clearly
provide your evaluation scores within <answer> </answer> tags, i.e., for
example,<answer>3</answer><answer>5</answer>
    """,
    system_template_point="""You are a helpful assistant. The assistant first performs a detailed,
step-by-step reasoning process in its mind and then provides the user with
the answer. The reasoning process and answer are enclosed within  and <answer> </answer> tags, respectively, i.e., <answer> answer here </answer>. Now the user asks you
to judge the performance of an AI assistant's response to the question.
Score the assistant 1-10 (higher=better). Criteria includes helpfulness,
relevance, accuracy, and level of detail. Avoid length, style or
other bias. After thinking, when you finally reach a conclusion, clearly
provide your evaluation score within <answer> </answer> tags, i.e., for
example,<answer>7</answer>
    """,
    user_template_list="""
    [Question]
{question}
[Assistant 1's Answer]
{answer_a}
[Assistant 2's Answer]
{answer_b}""",
    user_template_point="""
    [Question]
{question}
[Assistant's Answer]
{answer}
""",
)
