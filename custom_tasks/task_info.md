# Evaluation Tests

This folder contains a suite of tests designed to evaluate the performance and capabilities of a language model (LLM). The tests are towards the domain of e-commerce scenarios for the sole purpose of demonstration, but could be extended to other domains and use cases. The tests are categorized into different domains, each focusing on specific aspects of language understanding, reasoning, and response generation.

We will be covering thw following topics:

1. How to write an evaluation test yaml file

2. List of tasks

## Writing an evaluation test yaml file

Let's use the example of summarization capability as an example. The goal of the test is to use the language model to convert a product specs into a description in paragraph format.

```
- case_id: 001
  category: Summarization
  sub_category: Product Information Understanding and Generation
  system_prompt: >
    You are a product description expert helping to generate concise, customer-friendly product descriptions in standard paragraph form based on given technical specifications. 
    Ensure that the product description is clear, highlights key features, and is suitable for e-commerce platforms.
  instruction: >
    Generate a concise product description in paragraph format for the following product specifications: 
    '4K Ultra HD 65-inch TV, HDR10+, Dolby Vision, Smart TV with built-in Alexa, 3 HDMI ports, Wi-Fi enabled, Energy Star certified.'
  expected_response: >
    Experience stunning picture quality with this 65-inch 4K Ultra HD Smart TV, featuring HDR10+ and Dolby Vision for vibrant, lifelike visuals. 
    With built-in Alexa, enjoy hands-free control, and easily connect devices through 3 HDMI ports. Energy Star certified for energy efficiency."
  potential_challenges: null
  difficulty_level: easy
```

- `case_id`: A unique identifier for the test case. This helps track and organize different test cases within the evaluation.

- `category`: The general task category being tested, here it is “Summarization”, indicating that the test is related to summarizing information.

- `sub_category`: A more specific classification within the category, which in this case is "Product Information Understanding and Generation," focusing on summarizing product details.

- `system_prompt`: The initial setup or role assigned to the candidate model, giving context for the task (e.g., generating product descriptions).

- `instruction`: A specific task for the candidate model, detailing what needs to be done based on provided inputs (e.g., creating a description from technical specs).

- `expected_response`: A sample ideal response that the candidate model should generate, giving evaluators a reference for correct output.

- `potential_challenges`: Identifies potential difficulties or common errors the candidate model might face, though it's null here. This information could be provided to the judge-LLM as hints about what to evaluate in the candidate model's reponse.

- `difficulty_level`: Describes the expected difficulty level of the task, useful for designing a well-rounded evaluation set. The value can be "easy", "medium", "hard". When analysing the overall performance of the candidate model on the task, you might consider 2 approaches: 1) a flat average of the scores of all cases, 2) a weighted average where the weight is a function of the difficulty level.

## Code to run the task

With each test yaml file in `custom_tasks/` directory, there is a python file associated with in `task_handler/` directory.
Within the python file, there is a runner function.

Below is the runner function for the summarization task

```
# sorted list of fields in the original test case, a dict
SORTED_LEGACY_KEYS = ["case_id", 
               "category", 
               "sub_category", 
               "system_prompt", 
               "instruction",
               "expected_response",
               "potential_challenges",
               "difficulty_level"]
# sorted list of fields added to the case after inference with the candidate model
SORTED_NEW_KEYS = ["response_candidate_model"]
# sorted list of fields added to the case after evaluation with judge-model
SORTED_EVAL_KEYS = ["score", "judge_feedback"]


def run_summarization_task(llm: Any, 
                           dataset: List[Dict[str, Any]], 
                           **kwargs
                           ):
    """
    Executes a summarization task on a given dataset using 
    a specified language model.

    Args:
        llm (Any): The model under assessment, responsible 
        for generating text outputs.
        dataset (List[Dict[str, Any]]): A dataset where 
        each record is a dictionary containing task-specific 
        fields.
        **kwargs: Additional configuration parameters.

    Returns:
        List[Dict[str, Any]]: The updated dataset with model 
        responses.
    """
    # 
    validate_test_dataset(dataset, required_keys=SORTED_LEGACY_KEYS)

    prompt_template = ChatPromptTemplate.from_messages(
        [("system", "{system_prompt}"), 
         ("user", "{instruction}")])
    chain_llm = prompt_template | llm
    all_fields = SORTED_LEGACY_KEYS +\
                 SORTED_NEW_KEYS +\
                 SORTED_EVAL_KEYS

    for ix, record in enumerate(tqdm(dataset)):
        system_prompt = dedent(record["system_prompt"])
        user_query = dedent(record["instruction"])
        out = chain_llm.invoke(
            {"system_prompt": system_prompt, 
             "instruction": user_query})
            
        record["response_candidate_model"] = out
        # Append eval keys
        for fld in SORTED_EVAL_KEYS:
            record[fld] = None
        
        # sort keys
        dataset[ix] = custom_sort(record, all_fields)

    return dataset
    ```

## List of tests

- `custom_tasks/summarization_results.yaml`
- `custom_tasks/instruction_following_ecommerce_tests.yaml`
- `custom_tasks/inference_speed_ecommerce_tests.yaml`
- `custom_tasks/ethical_reasoning_ecommerce_tests.yaml`
- `custom_tasks/general_knowledge_ecommerce_tests.yaml`


### Summarization

- **Objective**: To assess the model's ability to generate concise and coherent product descriptions based on given technical specifications.

- **Implementation**:
  - Provide the model with lengthy text passages and assess the quality of its summaries, or its ability to convert bullet points text to paragraph format text. 
  - Use a `run_summarization_test` function to generate responses from the candidate model.
  - Requires a `summarization_test_set.yaml` file with various text examples for testing different summary lengths and complexities.

### Instruction Following

- **Objective**: To evaluate how well the model follows specific instructions in customer interaction scenarios, particularly in an e-commerce context. In that particular case, the system prompt includes details on the company policy for returning products. Based on the query of the user, that information is used by the judge LLM to answer the customer.

- **Implementation**:
  - Create a series of interconnected prompts to be stored in `instruction_following_ecommerce_tests.yaml`
  - Use a `run_instruction_following_task` function to generate responses from the candidate model.

### General knowledge

- **Objective**: To evaluate the model's ability to provide accurate and detailed product information and technical definitions.

- **Implementation**:
  -  Each test case includes a system prompt, an instruction asking for information or comparisons, and an expected response. The model's knowledge and clarity in explaining concepts are tested.
  - Use a `run_general_knowledge_task` function to generate responses from the candidate model.

### Context Retention

- **Objective**:Test the model's ability to retain and accurately apply information from earlier parts of a conversation or text in later responses.

- **Implementation**:
  - Provide a series of prompts where important information is introduced early and referenced in later prompts.
  - Use a `run_context_retention_test` function to evaluate the model’s consistency and accuracy in recalling and applying the earlier information.
  - Requires a `context_retention_test_set.yaml` file with structured sequences of prompts that progressively build on prior context.

### Inference speed

- **Objective**: To measure the model's response time when generating outputs of varying lengths, assessing the correlation between response length and latency.

- **Implementation**:
  - Each test case specifies a system prompt, an instruction, and an expected output length. The model's speed and efficiency in generating responses are evaluated.
  - Use a `run_inference_speed_task` function to generate responses from the candidate model.


## Potential  Evaluation Ideas

Various other tests can be developed to evaluate a language model's capabilities across a wide range of tasks. Below are a few examples:

- **Multilingual Proficiency**: Test understanding and response generation in multiple languages.
- **Abstraction and Generalization**: Assess the application of learned concepts to new scenarios.
- **Emotional Intelligence**: Evaluate recognition and response to emotional cues.
- **Code Generation and Debugging**: Test proficiency in writing and debugging code.
- **Multimodal Understanding**: If applicable, assess reasoning across text and images.
- **Adversarial Robustness**: Test performance against adversarial inputs.
- **Bias Detection and Mitigation**: Evaluate bias recognition and mitigation.
- **Task Adaptation**: Assess adaptation to new tasks with minimal instruction.
- **Long-term Dependency Handling**: Test coherence in long contexts.
- **Consistency Testing**: Evaluate the model's consistency in responses to similar questions.

