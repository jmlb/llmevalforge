from tqdm import tqdm
from langchain.schema import HumanMessage, SystemMessage


def run_consistency_test(model, dataset):
    results = []
    for case in tqdm(dataset, desc="Running consistency tests"):
        messages = [SystemMessage(content="You are a helpful AI assistant.")]
        
        for i, exchange in enumerate(case['conversation']):
            messages.append(HumanMessage(content=exchange['user']))
            response = model(messages)
            messages.append(SystemMessage(content=response.content))
            
            if 'eval_point' in exchange:
                eval_result = evaluate_consistency(response.content, exchange['eval_point'])
                
                results.append({
                    'case_id': case['id'],
                    'exchange_number': i + 1,
                    'user_input': exchange['user'],
                    'model_response': response.content,
                    'eval_point': exchange['eval_point'],
                    'score': eval_result['score'],
                    'explanation': eval_result['explanation']
                })
    
    return results

def evaluate_consistency(response, eval_point):
    # Implement consistency evaluation logic here
    # This is a placeholder implementation
    if eval_point.lower() in response.lower():
        return {'score': 1, 'explanation': 'The model provided a consistent response.'}
    else:
        return {'score': 0, 'explanation': 'The model\'s response was inconsistent.'}