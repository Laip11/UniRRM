"""m-prometheus template."""

from ._base import EvalPromptTemplate, make_extractor

m_prometheus = EvalPromptTemplate(
    template_name="m-prometheus",
    formatted_prompt=None,
    listwise_answer_extractor=make_extractor(r'\[RESULT\]\s*([A-Za-z])'),
    system_template_list="You are a fair judge assistant assigned to deliver insightful feedback that compares individual performances, highlighting how each stands relative to others within the same cohort.",
    user_template_list="""
###Task Description: 
An instruction (might include an Input inside it), two responses to evaluate (denoted as Response A and Response B), a reference answer(may still not include everything), and an evaluation criteria are given. 
1. Write a detailed feedback that assess the quality of the two responses strictly based on the given evaluation criteria, not evaluating in general. 
2. Make comparisons between Response A, Response B, and the Reference Answer. Instead of examining Response A and Response B separately, go straight to the point and mention about the commonalities and differences between them. 
3. After writing the feedback, indicate the better response, either "A" or "B". 
4. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (Either "A" or "B")" 
5. Please do not generate any other opening, closing, and explanations.

###The instruction to evaluate:
{question}

##Response A to evaluate:
{answer_a}

##Response B to evaluate:
{answer_b}

###Evaluation Criteria: 
Is the response structured to promote readability and coherence? 
Does the response exhibit excellent organization?

##Feedback:
""",
)
