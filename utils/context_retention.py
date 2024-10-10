from tqdm import tqdm
from langchain.schema import HumanMessage, SystemMessage

def run_context_retention_test(model, dataset):
    results = []
    for case in tqdm(dataset, desc="Running context retention tests"):
        messages = [SystemMessage(content="You are a helpful AI assistant.")]
        context_score = 0
        
        for i, exchange in enumerate(case['conversation']):
            messages.append(HumanMessage(content=exchange['user']))
            response = model(messages)
            messages.append(SystemMessage(content=response.content))
            
            if 'eval_point' in exchange:
                eval_result = evaluate_response(response.content, exchange['eval_point'])
                context_score += eval_result['score']
                
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

def evaluate_response(response, eval_point):
    if eval_point.lower() in response.lower():
        return {'score': 1, 'explanation': 'The model correctly used the context.'}
    else:
        return {'score': 0, 'explanation': 'The model failed to use the relevant context.'}