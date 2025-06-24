from math_datasets.datasets import Dataset, GSM8K, SVAMP
from math_datasets.generators import generate_responses, GeminiGenerate, TransformersGenerate, OllamaGenerate
from math_datasets.fine_tuning.llm.transformer_llm import TransformerLLM
from math_datasets.evaluator import evaluate_all
import os
from dotenv import load_dotenv
from typing import List, Literal
from pathlib import Path
import argparse

load_dotenv(override=True)

SAVE_DIR = Path(__file__).parent

# python inference/direct_prompting_test/main.py --first-n 100 && python inference/rewoo_test/main.py --first-n 100
GEMINI_MODELS = [
    "gemini-2.0-flash",
    "gemma-3-27b-it",
]

TRANSFORMERS_MODELS = [
    "HuggingFaceTB/SmolLM2-360M-Instruct",
    "Qwen/Qwen2.5-0.5B-Instruct",
    # "Qwen/Qwen3-0.6B",
    # "HuggingFaceTB/SmolLM2-1.7B-Instruct",
]


OLLAMA_MODELS = [
    "qwen3:0.6b",
    "qwen3:1.7b",
    # "qwen2.5-coder:0.5b",
    # "qwen2.5:0.5b",
    # "qwen2-math:1.5b",

    # "deepseek-r1:1.5b",
    
    # "gemma3:1b",
    
    # "llama3.2:1b",
    
    # "granite3.3:2b",
    # "tinyllama:1.1b",
    # "smollm2:135m",
    # "smollm2:360m",
    # "smollm2:1.7b",

]

def get_ollama_model_name_identifer(model_name: str) -> str:
    return "ollama/" + model_name



def generate_responses_for_gemini_models(datasets: List[Dataset], model_names: List[str], first_n: int|None=None, dataset_split: Literal["test", "train"]="test"):
    for model_name in model_names:
        for dataset in datasets:
            generate_responses(
                dataset, 
                model_name=model_name, 
                generator=GeminiGenerate(model_name=model_name, wait_frequency=15), 
                save_dir=SAVE_DIR.as_posix(), 
                first_n=first_n,
                dataset_split=dataset_split
            )
            dataset.clear_cache()

def generate_responses_for_ollama_models(datasets: List[Dataset], model_names: List[str], first_n: int|None=None, dataset_split: Literal["test", "train"]="test"):
    for model_name in model_names:
        for dataset in datasets:
            generate_responses(
                dataset, 
                model_name=get_ollama_model_name_identifer(model_name), 
                generator=OllamaGenerate(model_name=model_name), 
                save_dir=SAVE_DIR.as_posix(), 
                first_n=first_n,
                dataset_split=dataset_split
            )
            dataset.clear_cache()

def generate_responses_for_transformer_models(datasets: List[Dataset], model_names: List[str], first_n: int|None=None, dataset_split: Literal["test", "train"]="test"):
    for model_name in model_names:
        for dataset in datasets:
            generate_responses(
                dataset, 
                model_name=model_name, 
                generator=TransformersGenerate(model=TransformerLLM(model_name=model_name)),
                save_dir=SAVE_DIR, 
                first_n=first_n,
                dataset_split=dataset_split
            )
            dataset.clear_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model performance before and after training")
    parser.add_argument("--first-n", type=int, default=None,
                       help="Number of samples to evaluate (default: None, which means all samples will be evaluated)")
    args = parser.parse_args()

    datasets = [SVAMP, GSM8K]
    first_n = args.first_n

    print("Generating responses for Gemini models...")
    # generate_responses_for_gemini_models(datasets, GEMINI_MODELS, first_n=first_n, dataset_split="test")
    generate_responses_for_ollama_models(datasets, OLLAMA_MODELS, first_n=first_n, dataset_split="test")

    print("Generating responses for Transformers models...")
    # generate_responses_for_transformer_models(datasets, TRANSFORMERS_MODELS, first_n=first_n, dataset_split="test")
    ollama_names = [get_ollama_model_name_identifer(model_name) for model_name in OLLAMA_MODELS]
    # Evaluate all datasets
    df = evaluate_all(GEMINI_MODELS + TRANSFORMERS_MODELS + ollama_names, datasets, save_dir=SAVE_DIR, use_transformated_answers=False, use_first_n=first_n)
    print(df)
    df.to_csv(SAVE_DIR / "dp_evaluation.csv", index=False)
