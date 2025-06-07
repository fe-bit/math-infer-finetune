from math_datasets.datasets import Dataset, GSM8K, SVAMP
from math_datasets.generators import generate_responses, OllamaGenerate, GeminiGenerate, TransformersGenerate
from math_datasets.evaluator import evaluate_all
from math_datasets.training_data import get_training_data
import os
from dotenv import load_dotenv
from typing import List, Literal
from pathlib import Path

load_dotenv(override=True)

SAVE_DIR = Path(__file__).parent.as_posix()


OLLAMA_MODELS = [
    # "smollm2:135m",
    # "smollm2:360m",
    # "qwen2.5:0.5b",
    # "qwen3:0.6b",
    # "llama3.2:1b",
    # "qwen2-math:1.5b",
    # "smollm2:1.7b",
    # "qwen3:1.7b",
    # "qwen3:4b",
    # "mistral:7b",
]

GEMINI_MODELS = [
    "gemini-2.0-flash",
    # "gemma-3-27b-it",
]

TRANSFORMERS_MODELS = [
    # "HuggingFaceTB/SmolLM2-1.7B-Instruct",
]

def generate_responses_for_ollama_models(datasets: List[Dataset], model_names: List[str], first_n: int|None=None, dataset_split: Literal["test", "train"]="test"):
    for dataset in datasets:
        for model_name in model_names:
            generator = OllamaGenerate(model_name=model_name)
            generate_responses(
                dataset, 
                model_name=model_name, 
                generator=generator, 
                save_dir=SAVE_DIR, 
                first_n=first_n,
                dataset_split=dataset_split
            )
        
        dataset.clear_cache()


def generate_responses_for_gemini_models(datasets: List[Dataset], model_names: List[str], first_n: int|None=None, dataset_split: Literal["test", "train"]="test"):
    for model_name in model_names:
        for dataset in datasets:
            generate_responses(
                dataset, 
                model_name=model_name, 
                generator=GeminiGenerate(model_name=model_name, wait_frequency=5), 
                save_dir=SAVE_DIR, 
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
                generator=TransformersGenerate(model_name=model_name),
                save_dir=SAVE_DIR, 
                first_n=first_n,
                dataset_split=dataset_split
            )
            dataset.clear_cache()

if __name__ == "__main__":
    datasets = [GSM8K, SVAMP]
    first_n = 450
    # Generate responses for Ollama models
    # print("Generating responses for Ollama models...")
    # generate_responses_for_ollama_models(datasets, OLLAMA_MODELS, first_n=first_n, dataset_split="train")
    
    print("Generating responses for Gemini models...")
    # Generate responses for Gemini models
    generate_responses_for_gemini_models(datasets, GEMINI_MODELS, first_n=first_n, dataset_split="train")

    # print("Generating responses for Transformers models...")
    # generate_responses_for_transformer_models(datasets, TRANSFORMERS_MODELS, first_n=first_n, dataset_split="train")
    
    # Evaluate all datasets
    df = evaluate_all(OLLAMA_MODELS + GEMINI_MODELS, datasets, save_dir=SAVE_DIR, use_transformated_answers=False)
    print(df)

    df_train1 = get_training_data(GEMINI_MODELS[0], datasets[0], SAVE_DIR)
    df_train2 = get_training_data(GEMINI_MODELS[0], datasets[1], SAVE_DIR)

    df_train = df_train1.add(df_train2)
    df_train.to_csv(Path(SAVE_DIR) / "train_data.csv", index=False)
