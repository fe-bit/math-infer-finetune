from math_datasets.datasets import Dataset, GSM8K, SVAMP
from math_datasets.generators import generate_responses
from math_datasets.evaluator import evaluate_all
from dotenv import load_dotenv
from typing import List, Literal
from pathlib import Path
from math_datasets.generators import Generate, generate_responses, TransformersGenerate
from math_datasets.fine_tuning.llm import TransformerLLM
import time
from rewoo import ReWOOGeminiModel
from rewoo_local import ReWOOLocalModel

load_dotenv(override=True)

SAVE_DIR = Path(__file__).parent.as_posix()

GEMINI_MODELS = [
    "gemini-2.0-flash",
]

TRANSFORMER_MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
]

class ReWOOGeminiModelGenerate(Generate):
    def __init__(self, rewoo_model: ReWOOGeminiModel|ReWOOLocalModel, sleep_time: int=5):
        self.rewoo_model = rewoo_model
        self.sleep_time = sleep_time

    def generate(self, prompt, entry: dict[str, str]={}) -> str:
        counter = 0
        while True:
            try:
                time.sleep(self.sleep_time)
                resp = self.rewoo_model(prompt)
                entry["model_history"] = resp
                return resp[-1]["solve"]["result"]
            except Exception as e:
                print(f"Error: {e}")
                t = 300
                print(f"Retrying in {t + self.sleep_time} seconds...")
                counter += 1
                if counter > 5:
                    entry["model_history"] = "Error occured."
                    return "Error occured."
                print("Counter:", counter)
                time.sleep(t + self.sleep_time)


def generate_responses_for_gemini_models(datasets: List[Dataset], model_names: List[str], first_n: int|None=None, dataset_split: Literal["test", "train"]="test"):
    for model_name in model_names:
        for dataset in datasets:
            generate_responses(
                dataset, 
                model_name=model_name, 
                generator=ReWOOGeminiModelGenerate(ReWOOGeminiModel(model_name=model_name, sleep_time=15), sleep_time=30), 
                save_dir=SAVE_DIR, 
                first_n=first_n,
                dataset_split=dataset_split
            )
            dataset.clear_cache()


def generate_responses_for_local_models(datasets: List[Dataset], model_names: List[str], first_n: int|None=None, dataset_split: Literal["test", "train"]="test"):
    for model_name in model_names:
        for dataset in datasets:
            generate_responses(
                dataset, 
                model_name=model_name, 
                generator=ReWOOGeminiModelGenerate(ReWOOLocalModel(llm=TransformersGenerate(TransformerLLM(model_name))), sleep_time=0), 
                save_dir=SAVE_DIR, 
                first_n=first_n,
                dataset_split=dataset_split
            )
            dataset.clear_cache()

if __name__ == "__main__":
    datasets = [SVAMP, GSM8K]
    first_n = 100
    
    # generate_responses_for_gemini_models(datasets, GEMINI_MODELS, first_n=first_n, dataset_split="test")
    generate_responses_for_local_models(datasets, TRANSFORMER_MODELS, first_n=first_n, dataset_split="test")
    
    df = evaluate_all(GEMINI_MODELS + TRANSFORMER_MODELS, datasets, save_dir=SAVE_DIR, use_transformated_answers=False)
    print(df)
