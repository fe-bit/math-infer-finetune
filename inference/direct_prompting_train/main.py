from math_datasets.datasets import Dataset, GSM8K, SVAMP
from math_datasets.generators import generate_responses, OllamaGenerate, GeminiGenerate
from math_datasets.evaluator import evaluate_all
from math_datasets.training_data import get_training_data
import os
from dotenv import load_dotenv
from typing import List, Literal
from pathlib import Path

load_dotenv(override=True)

SAVE_DIR = Path(__file__).parent.as_posix()

GEMINI_MODELS = [
    "gemini-2.0-flash",
]


def generate_responses_for_gemini_models(datasets: List[Dataset], model_names: List[str], first_n: int|None=None, dataset_split: Literal["test", "train"]="test"):
    for model_name in model_names:
        for dataset in datasets:
            generate_responses(
                dataset, 
                model_name=model_name, 
                generator=GeminiGenerate(model_name=model_name, wait_frequency=15), 
                save_dir=SAVE_DIR,
                first_n=first_n,
                dataset_split=dataset_split,
            )
            dataset.clear_cache()

def generate_responses_for_transformer_models(datasets: List[Dataset], model_names: List[str], first_n: int|None=None, dataset_split: Literal["test", "train"]="test"):
    for model_name in model_names:
        for dataset in datasets:
            generate_responses(
                dataset, 
                model_name=model_name, 
                generator=TransformersGenerate(model_name=model_name),
                save_dir=SAVE_DIR, 
                first_n=first_n,
                dataset_split=dataset_split
            )
            dataset.clear_cache()

if __name__ == "__main__":
    datasets = [GSM8K, SVAMP]
    first_n = None

    print(len(GSM8K.get_dataset()["train"]))
    print(len(SVAMP.get_dataset()["train"]))

    print("Generating responses for Gemini models...")
    generate_responses_for_gemini_models(datasets, GEMINI_MODELS, first_n=first_n, dataset_split="train")

    # Evaluate all datasets
    df = evaluate_all(GEMINI_MODELS, datasets, save_dir=SAVE_DIR, use_transformated_answers=False)
    print(df)
