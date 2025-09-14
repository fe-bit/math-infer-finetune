from math_datasets.datasets import Dataset, GSM8K, SVAMP
from math_datasets.generators import generate_responses
from math_datasets.evaluator import evaluate_all, evaluate_detail
from dotenv import load_dotenv
from typing import List, Literal
from pathlib import Path
from math_datasets.generators import generate_responses, TransformersGenerate, GeminiGenerate, OllamaGenerate
from math_datasets.fine_tuning.llm import TransformerLLM
import argparse
import torch
import os
from math_datasets.generators.reflection import ReflectionGenerate
from langchain_ollama import ChatOllama


load_dotenv(override=True)

SAVE_DIR = Path(__file__).parent
RESULTS_DIR = Path(__file__).parent.parent.parent / "results" / "dp-reflect"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


GEMINI_MODELS = [
    # "gemini-2.0-flash",
    # "gemma-3-27b-it",
]

TRANSFORMER_MODELS = [
    # "Qwen/Qwen2.5-0.5B-Instruct",
    # "Qwen/Qwen3-0.6B",
    # "HuggingFaceTB/SmolLM2-135M-Instruct",
]

OLLAMA_MODELS = [
    # "smollm2:135m",
    "smollm2:360m",
    "qwen2.5:0.5b",
    "qwen3:0.6b",
    # "llama3.2:1b",
    # "gemma3:1b",
    # "qwen2-math:1.5b",
    # "qwen2.5:1.5b",
    "deepseek-r1:1.5b",
    "qwen3:1.7b",
    # "smollm2:1.7b",
    # "qwen3:4b",
]

def get_model_name_identifer(model_name: str, fine_tuned: bool) -> str:
    if fine_tuned:
        return model_name + "/after_training"
    else:
        return model_name + "/before_training"
    
def get_ollama_model_name_identifer(model_name: str) -> str:
    return "ollama/" + model_name.replace(":", "_")

def generate_responses_for_gemini_models(datasets: List[Dataset], model_names: List[str], first_n: int|None=None, dataset_split: Literal["test", "train"]="test"):
    for model_name in model_names:
        generator = GeminiGenerate(model_name=model_name, wait_frequency=0)
        for dataset in datasets:
            generate_responses(
                dataset, 
                model_name=model_name, 
                generator=generator, 
                save_dir=SAVE_DIR.as_posix(), 
                first_n=first_n,
                dataset_split=dataset_split,
                overwrite=False
            )
            df = evaluate_detail(model_name, dataset=dataset, use_transformated_answers=False, additional_metrics=GeminiGenerate, save_dir=SAVE_DIR.as_posix(), use_first_n=first_n)
            f = RESULTS_DIR / dataset.name / "google" / f"{model_name}.csv"
            f.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(f, index=False)

def generate_responses_for_ollama_models(datasets: List[Dataset], model_names: List[str], first_n: int|None=None, dataset_split: Literal["test", "train"]="test"):
    for model_name in model_names:
        generator = ReflectionGenerate.init_ollama(model_name)
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
            df = evaluate_detail(get_ollama_model_name_identifer(model_name), dataset=dataset, use_transformated_answers=False, additional_metrics=ReflectionGenerate, save_dir=SAVE_DIR.as_posix(), use_first_n=first_n)
            f = RESULTS_DIR / dataset.name / f"{get_ollama_model_name_identifer(model_name).replace(":", "_")}.xlsx"
            f.parent.mkdir(parents=True, exist_ok=True)
            df.to_excel(f, index=False, sheet_name="Original")


def main(first_n: int|None=None):
    datasets = [SVAMP, GSM8K]
    generate_responses_for_gemini_models(datasets, GEMINI_MODELS, first_n=first_n, dataset_split="test")
    generate_responses_for_ollama_models(datasets, OLLAMA_MODELS, first_n=first_n, dataset_split="test")
   
    all_model_names = GEMINI_MODELS + \
        [get_ollama_model_name_identifer(model_name) for model_name in OLLAMA_MODELS]
    
    df = evaluate_all(all_model_names, datasets, save_dir=SAVE_DIR.as_posix(), use_transformated_answers=False, use_first_n=first_n)
    print(df)
    df.to_csv(SAVE_DIR / "direct_prompting_evaluation.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model performance before and after training")
    parser.add_argument("--model-name", type=str, required=False, 
                       help="Name of the model to evaluate. If not provided, all models will be evaluated.")
    parser.add_argument("--first-n", type=int, default=None,
                       help="Number of samples to evaluate (default: None, which means all samples will be evaluated)")
    parser.add_argument("--resume", action="store_true", default=False,
                       help="Whether to overwrite or resume the evaluation results for the trained model")
    args = parser.parse_args()

    datasets = [SVAMP, GSM8K]
    first_n = args.first_n
    if args.model_name is None:
        # generate_responses_for_gemini_models(datasets, GEMINI_MODELS, first_n=first_n, dataset_split="test")
        generate_responses_for_ollama_models(datasets, OLLAMA_MODELS, first_n=first_n, dataset_split="test")
        # generate_responses_for_local_models_before_fine_tuning(datasets, TRANSFORMER_MODELS, first_n=first_n, dataset_split="test")
        # generate_responses_for_local_models_after_fine_tuning(datasets, TRANSFORMER_MODELS, first_n=first_n, dataset_split="test")
        
        all_model_names = GEMINI_MODELS + \
            [get_model_name_identifer(model_name, fine_tuned=False) for model_name in TRANSFORMER_MODELS] + \
            [get_model_name_identifer(model_name, fine_tuned=True) for model_name in TRANSFORMER_MODELS] + \
            [get_ollama_model_name_identifer(model_name) for model_name in OLLAMA_MODELS]
        
        df = evaluate_all(all_model_names, datasets, save_dir=SAVE_DIR.as_posix(), use_transformated_answers=False, use_first_n=first_n)
        print(df)
        df.to_csv(SAVE_DIR / "direct_prompting_evaluation.csv", index=False)
    else:
        models = [args.model_name]
         
        all_model_names = GEMINI_MODELS + \
            [get_model_name_identifer(model_name, fine_tuned=False) for model_name in models] + \
            [get_model_name_identifer(model_name, fine_tuned=True) for model_name in models] + \
            [get_ollama_model_name_identifer(model_name) for model_name in OLLAMA_MODELS]
        df = evaluate_all(all_model_names, datasets, save_dir=SAVE_DIR.as_posix(), use_transformated_answers=False)
        print(df)
        os.makedirs(SAVE_DIR / f"{args.model_name}", exist_ok=True)
        df.to_csv(SAVE_DIR / f"{args.model_name}_dp_evaluation.csv", index=False)

