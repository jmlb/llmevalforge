import logging
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class Evaluator:
    """Handles evaluation of model outputs."""
    
    def __init__(self, model: Any, prompt_template: str):
        """Initialize with evaluation model."""
        self.model = model
        self.prompt_template = prompt_template
    
    def evaluate_response(
        self,
        system_prompt: str,
        instruction: str,
        response: str,
        expected_response: str,
        challenges: str
    ) -> Dict[str, Any]:
        """
        Evaluate a single response.
        
        Args:
            system_prompt: system_prompt for the evaluator
            instruction: instruction to the candidate model
            response: Candidate Model's response to evaluate
            expected_response: Expected response
            challenges: Additional context for evaluation (potential challenges for the candidate model)
            
        Returns:
            Evaluation results including score and feedback
        """
        # Create evaluation prompt
        prompt = self.prompt_template.replace("[[system_prompt_candidate_model]]", system_prompt)
        prompt = prompt.replace("[[instruction_candidate_model]]", instruction)
        prompt = prompt.replace("[[response_candidate_model]]", response)
        prompt = prompt.replace("[[expected_response_candidate_model]]", expected_response)
        prompt = prompt.replace("[[challenges]]", challenges)
        try:
            # Get evaluation from model
            eval_response = self.model.invoke(prompt)
            eval_text = eval_response.content if hasattr(eval_response, 'content') else eval_response
            
            # Extract score and feedback
            score = self._extract_score(eval_text)
            
            return {
                "score": score,
                "feedback": eval_text
            }
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return {
                "score": 0,
                "feedback": f"Evaluation error: {str(e)}"
            }
    
    def _extract_score(self, eval_text: str) -> float:
        """Extract numerical score from evaluation text."""
        import re
        
        # Look for score in format "Score: X" or similar
        patterns = [
            r'score:\s*(\d+(?:\.\d+)?)',
            r'\*\*score\*\*:\s*(\d+(?:\.\d+)?)'
        ]
        
        for pattern in patterns:
            if match := re.search(pattern, eval_text.lower()):
                return float(match.group(1))
        
        logger.warning(f"Could not extract score from: {eval_text}")
        return 0.0
    
    def evaluate_results(
        self,
        records: List[Dict[str, Any]],
        output_path: Optional[Path] = None
    ) -> List[Dict[str, Any]]:
        """
        Evaluate a list of records.
        
        Args:
            records: List of task records to evaluate
            output_path: Optional path to save evaluation results
            
        Returns:
            List of records updated with evaluation scores and feedback from the evaluator
        """
        evaluated_results = []
        
        for record in tqdm(records, desc="Evaluating responses"):
            try:
                evaluation = self.evaluate_response(
                    instruction=record["instruction"],
                    response=record["model_response"],
                    expected_response=record["expected_response"],
                    system_prompt=record["system_prompt"],
                    challenges=record.get("challenges", "")
                )
                
                # Update result with evaluation
                record.update({
                    "score": evaluation["score"],
                    "feedback": evaluation["feedback"]
                })
                
            except Exception as e:
                logger.error(f"Failed to evaluate result: {e}")
                record.update({
                    "score": 0,
                    "feedback": f"Evaluation error: {str(e)}"
                })
            
            evaluated_results.append(record)
        
        # # Save results if path provided
        # if output_path:
        #     try:
        #         Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        #         with open(output_path, 'w') as f:
        #             json.dump(evaluated_results, f, indent=2)
        #     except Exception as e:
        #         logger.error(f"Failed to save evaluation results: {e}")
        
        return evaluated_results


if __name__ == "__main__":
    # Example usage
    from models import create_model
    
    # Create evaluator
    eval_model = create_model("gpt-4", model_type="openai")
    eval_prompt = "Evaluate the response with respect to the expected response"
    evaluator = Evaluator(eval_model, eval_prompt)

    # Test evaluation: the output is the record updated wit hscore and evaluator feedbacks
    result = evaluator.evaluate_response(
        system_prompt="act as a AI assistant",
        instruction="Paraphrase the sentence: A cat on a mat", 
        response="The cat is on the mat.",
        expected_response="A cat is sitting on a mat.",
        challenges="The response must refer to the cat."
    )
    
    print("Evaluation result:", result)
