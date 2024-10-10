from tqdm import tqdm
from langchain.schema import HumanMessage, SystemMessage

def run_fact_checking_test(model, dataset):
    results = []
    for case in tqdm(dataset, desc="Running fact-checking tests"):
        messages = [
            SystemMessage(content="You are an AI capable of fact-checking and distinguishing between factual information and opinions."),
            HumanMessage(content=case['input_prompt'])
        ]
        
        response = model(messages)
        
        eval_result = evaluate_fact_checking(response.content, case['long_response'])
        
        results.append({
            'case_id': case['id'],
            'input_prompt': case['input_prompt'],
            'model_response': response.content,
            'true_answer': case['long_response'],
            'score': eval_result['score'],
            'explanation': eval_result['explanation']
        })
    
    return results

def evaluate_fact_checking(response, true_answer):
    # Implement fact-checking evaluation logic here
    # This is a placeholder implementation
    score = 0
    if "fact" in true_answer.lower() and "fact" in response.lower():
        score += 5
    if "opinion" in true_answer.lower() and "opinion" in response.lower():
        score += 5
    
    return {
        'score': score,
        'explanation': f"Score: {score}/10. Placeholder explanation."
    }