from tqdm import tqdm
from textwrap import dedent
from langchain.schema import HumanMessage, SystemMessage

from utils.teacher_model_test import grade_student_response


SYSTEM_STUDENT_PROMPT = """
You are an AI capable of learning concepts and applying them to new scenarios.

"""

def run_abstraction_generalization_test(student_model, teacher_model, dataset):
    results = []
    for case in tqdm(dataset, desc="Running abstraction and generalization tests"):
        messages = [
            SystemMessage(content=SYSTEM_STUDENT_PROMPT),
            HumanMessage(content=case['instruction'])]
        response_str = student_model.invoke(messages)

        system_prompt = dedent("""
                                You are an AI assistant responsible for grading and providing detailed feedback based on the following three inputs: 
                                1) Instruction, 2) Student Response, and 3) Expected Response. 
                                
                                Your role is to:
                                1. Analyze the instruction to understand what is expected.
                                2. Compare the student response to the expected response, focusing on whether they agree semantically.
                                3. Assign a grade from 0 to 10, where a score of 10 means that the student response is fully aligned with the expected response semantically, and a score of 0 the 2 responses do not align.
                               
                               Your output must be organized in the following format:
                               score: [numerical value between 0 and 10]
                               feedback: [explanation of the score, highlighting the strengths and weaknesses of the student's response]

                                """)
        
        user_prompt = dedent(f"""
                            [INSTRUCTION]
                            {case["instruction"]}
                            [STUDENT RESPONSE]
                            {response_str}
                            [EXPECTED RESPONSE]
                            {case['long_response']}
                            """)
        eval_result = grade_student_response(teacher_model, system_prompt, user_prompt)
        case["score"] = eval_result['score']
        case["teacher_feedback"] = eval_result['feedback']
        case['student_response'] = response_str
        
        results.append(case)
    return results
