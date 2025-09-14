from math_datasets.datasets import Dataset, GSM8K, SVAMP
from math_datasets.generators import generate_responses
from math_datasets.evaluator import evaluate_all, evaluate_detail
from dotenv import load_dotenv
from typing import List, Literal
from pathlib import Path
from math_datasets.generators import generate_responses, ReWOORetryGenerate
from math_datasets.fine_tuning.llm import TransformerLLM
import argparse
import torch
import os

load_dotenv(override=True)

SAVE_DIR = Path(__file__).parent

RESULTS_DIR = Path(__file__).parent.parent.parent / "results" / "rewoo_retry"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

GEMINI_MODELS = [
    # "gemini-2.0-flash",
    # "gemma-3-27b-it",
]

TRANSFORMER_MODELS = [
    # "HuggingFaceTB/SmolLM2-360M-Instruct",
    # "Qwen/Qwen2.5-0.5B-Instruct",
    # "Qwen/Qwen3-0.6B",
]

OLLAMA_MODELS = [
    # "smollm2:135m",
    # "smollm2:360m",
    "qwen2.5:0.5b",
    "qwen3:0.6b",
    # "llama3.2:1b",
    # "gemma3:1b",
    # "qwen2-math:1.5b",
    # "qwen2.5:1.5b",
    "deepseek-r1:1.5b",
    "qwen3:1.7b",
    # "smollm2:1.7b",
]
    
def get_ollama_model_name_identifer(model_name: str) -> str:
    return "ollama/" + model_name.replace(":", "_")

def get_google_name_identifer(model_name: str) -> str:
    return "google/" + model_name


def generate_responses_for_gemini_models(datasets: List[Dataset], model_names: List[str], first_n: int|None=None, dataset_split: Literal["test", "train"]="test"):
    for model_name in model_names:
        generator = ReWOORetryGenerate.init_gemini(model_name=model_name, sleep_time=15)
        for dataset in datasets:
            generate_responses(
                dataset, 
                model_name=get_google_name_identifer(model_name), 
                generator=generator, 
                save_dir=SAVE_DIR.as_posix(), 
                first_n=first_n,
                dataset_split=dataset_split,
                overwrite=False
            )
            df = evaluate_detail(get_google_name_identifer(model_name), dataset=dataset, use_transformated_answers=False, additional_metrics=ReWOORetryGenerate, save_dir=SAVE_DIR.as_posix(), use_first_n=first_n)
            f = RESULTS_DIR / dataset.name / f"{get_google_name_identifer(model_name)}.xlsx"
            f.parent.mkdir(parents=True, exist_ok=True)
            df.to_excel(f, index=False)

def generate_responses_for_ollama_models(datasets: List[Dataset], model_names: List[str], first_n: int|None=None, dataset_split: Literal["test", "train"]="test"):
    for model_name in model_names:
        generator = ReWOORetryGenerate.init_ollama(model_name=model_name, with_examples=True)
        for dataset in datasets:
            generate_responses(
                dataset, 
                model_name=get_ollama_model_name_identifer(model_name), 
                generator=generator, 
                save_dir=SAVE_DIR.as_posix(), 
                first_n=first_n,
                dataset_split=dataset_split,
                overwrite=False
            )
            df = evaluate_detail(get_ollama_model_name_identifer(model_name), dataset=dataset, use_transformated_answers=False, additional_metrics=ReWOORetryGenerate, save_dir=SAVE_DIR.as_posix(), use_first_n=first_n)
            f = RESULTS_DIR / dataset.name / f"{get_ollama_model_name_identifer(model_name).replace(":", "_")}.xlsx"
            f.parent.mkdir(parents=True, exist_ok=True)
            df.to_excel(f, index=False, sheet_name="Original")
        del generator




def main(first_n: int|None=None):
    datasets = [SVAMP, GSM8K]
    generate_responses_for_gemini_models(datasets, GEMINI_MODELS, first_n=first_n, dataset_split="test")
    generate_responses_for_ollama_models(datasets, OLLAMA_MODELS, first_n=first_n, dataset_split="test")
    
    all_model_names = [get_google_name_identifer(model_name) for model_name in GEMINI_MODELS] + \
        [get_ollama_model_name_identifer(model_name) for model_name in OLLAMA_MODELS]

    df = evaluate_all(all_model_names, datasets, save_dir=SAVE_DIR.as_posix(), use_transformated_answers=False, use_first_n=first_n)
    print(df)
    df.to_csv(SAVE_DIR / "rewoo_evaluation.csv", index=False)
