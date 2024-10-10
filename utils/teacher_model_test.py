import re
from langchain.schema import HumanMessage, SystemMessage


def grade_student_response(teacher_model, system_prompt, user_prompt):
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    resp = teacher_model(messages).content
    
    score = resp.lower()
    score = score.strip().split("\n")[0]
    # Extract the grade (assuming it's an integer in the response)
    score = int(re.search(r'\d+', score).group())
    return {"score": score, "feedback": resp}

