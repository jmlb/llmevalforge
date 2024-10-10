from tqdm import tqdm
from langchain.schema import HumanMessage, SystemMessage

def run_basic_evaluation(model, dataset, configs):
    results = []
    for record in tqdm(dataset, desc="Running basic evaluation"):
        instruction = record["input_prompt"]
        true_answer = record["long_response"]
        
        messages = [
            SystemMessage(content=configs["student_model"]["system_prompt"]),
            HumanMessage(content=instruction)
        ]
        
        response = model(messages)
        
        results.append({
            "instruction": instruction,
            "true_answer": true_answer,
            "model_response": response.content,
            "category": record["category"],
            "difficulty": record["difficulty_level"]
        })
    
    return results
