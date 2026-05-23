"""Default scalar reward model template."""

from ._base import EvalPromptTemplate, make_extractor

default_scalar_rm = EvalPromptTemplate(
    template_name="scalar_rm",
    formatted_prompt=None,
    listwise_answer_extractor=make_extractor(r'\[([AB])\]'),
    user_template_list="",
)
