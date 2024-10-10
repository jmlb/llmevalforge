import time
import numpy as np
from tqdm import tqdm

def generate_input(token_count):
    words = ["the"] * (token_count // 2)
    return " ".join(words)

def measure_inference_time(model, input_text, max_tokens):
    start_time = time.time()
    _ = model([input_text])
    end_time = time.time()
    return end_time - start_time

def run_inference_speed_test(model, input_token_range=[10, 50, 100, 200, 500, 1000], output_token_range=[10, 50, 100, 200, 500], num_runs=5):
    results = []
    for input_tokens in tqdm(input_token_range, desc="Testing inference speed"):
        for output_tokens in output_token_range:
            input_text = generate_input(input_tokens)
            times = []
            for _ in range(num_runs):
                inference_time = measure_inference_time(model, input_text, output_tokens)
                times.append(inference_time)
            avg_time = np.mean(times)
            results.append({
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "avg_inference_time": avg_time
            })
    return results