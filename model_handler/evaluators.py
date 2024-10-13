import re
from tqdm import tqdm
from textwrap import dedent
from abc import ABC, abstractmethod
from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOpenAI


class BaseEvaluator(ABC):
    @abstractmethod
    def evaluate(self, student_response: str, expected_response: str, **kwargs) -> Dict[str, Any]:
        pass


class BaseScorer(ABC):
    @abstractmethod
    def score(self, evaluation_text: str) -> Any:
        pass


class NumericalScorer(BaseScorer):
    def score(self, evaluation_text: str) -> int:
        # Extract numerical score (implement the logic)
        out_lower = evaluation_text.lower()
        # Use a regular expression to find the grade
        match = re.search(r'\*\*grade\*\*:\s*(\d+)', out_lower)

        if match:
            grade = int(match.group(1))
            print(f"Extracted grade: {grade}")
        else:
            print("Grade not found.")
            grade = None
        
        return grade


class CategoryScorer(BaseScorer):
    def score(self, evaluation_text: str) -> str:
        # Extract category (implement the logic)
        pass


class ChatGPTEvaluator(BaseEvaluator):
    def __init__(self, model: ChatOpenAI, prompt_template: ChatPromptTemplate, scorer: BaseScorer):
        self.model = model
        self.prompt_template = prompt_template
        self.scorer = scorer

    def evaluate(self, student_response: str, expected_response: str, **kwargs) -> Dict[str, Any]:
        evaluation_prompt = self.prompt_template.format(
            student_response=student_response,
            expected_response=expected_response,
            **kwargs
        )
        
        evaluation_result = self.model(evaluation_prompt)
        score = self.scorer.score(evaluation_result)
        
        return {
            "score": score,
            "feedback": evaluation_result
        }


def run_evaluation(dataset: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
    
    try:
        model_params = kwargs['model_params']
    except ValueError as e:
        print(e)

    llm = ChatOpenAI(**model_params)

    user_prompt_template = dedent(kwargs.get('user_prompt', ""))
    system_prompt = dedent(kwargs.get('system_prompt', ""))
    prompt_template = ChatPromptTemplate.from_messages([("system", "{system_prompt}"), ("human", "{user_prompt}")])
    chain_llm = prompt_template | llm

    scorer_class = globals()[kwargs.get('scorer', 'NumericalScorer')]
    scorer = scorer_class()
    
    for ix, record in enumerate(tqdm(dataset)):
        user_prompt = user_prompt_template.format(student_system_prompt=record["system_prompt"],
                                                  student_instruction=record["instruction"], 
                                                  expected_response=record["expected_response"], 
                                                  student_response=record["response_candidate_model"]) 
        # Invoke the chain with the actual user instruction
        out = chain_llm.invoke({"system_prompt": system_prompt, "user_prompt": user_prompt})
        out = out.content
        record_score = scorer.score(out)
        record["score"] = record_score
        record["scorer_feedback"] = out
        print(record_score)
        dataset[ix] = record

    return dataset