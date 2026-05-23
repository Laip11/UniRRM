"""Skywork-Critic-Llama-3.1-70B template."""

from ._base import EvalPromptTemplate, make_extractor, make_pointwise_extractor

Skywork_Critic_Llama_3_1_70B = EvalPromptTemplate(
    template_name="Skywork_Critic_Llama_3_1_70B",
    system_template_list=None,
    formatted_prompt=None,
    listwise_answer_extractor=make_extractor(r'\[([AB])\]'),
    pointwise_answer_extractor=make_pointwise_extractor(r'总体[^\d]*(\d+)'),
    user_template_point="""
请扮演一个专业的对话质量评价专家，能够从多个维度诊断和分析 AI 助手对用户问题的回答，并进行总体打分（分值范围是 1-5）。你的评估应考虑回答的有用性、相关性、准确性、深度、创造性、安全性等众多维度，请注意，不同任务类型的指令对评估分析维度的侧重不一样，需要根据具体的问题进行分析。

[用户问题]
{instruction}

[助手回答的开始]
{response1}
[助手回答的结束]

你的详细评估和总体打分为: """,
    user_template_list="""Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user\'s instructions and answers the user\'s question better. 
Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. 
Please directly output your final verdict by strictly following this format: "[[A]]" if assistant A is better, "[[B]]" if assistant B is better.

[User Question]
{question}

[The Start of Assistant A's Answer]
{answer_a}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
{answer_b}
[The End of Assistant B's Answer]
""",
)
