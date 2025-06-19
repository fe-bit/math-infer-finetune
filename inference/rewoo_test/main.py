from math_datasets.datasets import Dataset, GSM8K, SVAMP
from math_datasets.generators import generate_responses
from math_datasets.evaluator import evaluate_all
from dotenv import load_dotenv
from typing import List, Literal
from pathlib import Path
from math_datasets.generators import generate_responses, ReWOOGenerate
from math_datasets.fine_tuning.llm import TransformerLLM
import argparse
import torch

load_dotenv(override=True)

SAVE_DIR = Path(__file__).parent

GEMINI_MODELS = [
    # "gemini-2.0-flash",
    "gemma-3-27b-it",
]

TRANSFORMER_MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
]

def get_model_name_identifer(model_name: str, fine_tuned: bool) -> str:
    if fine_tuned:
        return model_name + "/after_training"
    else:
        return model_name + "/before_training"


def get_checkpoint_path(model_name: str) -> Path:
    return SAVE_DIR / f"training-output/{model_name}"


def generate_responses_for_gemini_models(datasets: List[Dataset], model_names: List[str], first_n: int|None=None, dataset_split: Literal["test", "train"]="test"):
    for model_name in model_names:
        generator = ReWOOGenerate.init_gemini(model_name=model_name, sleep_time=15)
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
            dataset.clear_cache()


def generate_responses_for_local_models_before_fine_tuning(datasets: List[Dataset], model_names: List[str], first_n: int|None=None, dataset_split: Literal["test", "train"]="test"):
    for model_name in model_names:
        llm = TransformerLLM(model_name, dtype=torch.bfloat16)
        generator = ReWOOGenerate.init_transformer_llm(llm=llm, with_examples=True)
        for dataset in datasets:
            generate_responses(
                dataset, 
                model_name=get_model_name_identifer(model_name, fine_tuned=False), 
                generator=generator, 
                save_dir=SAVE_DIR.as_posix(), 
                first_n=first_n,
                dataset_split=dataset_split,
                overwrite=False
            )
            dataset.clear_cache()

def generate_responses_for_local_models_after_fine_tuning(datasets: List[Dataset], model_names: List[str], first_n: int|None=None, dataset_split: Literal["test", "train"]="test", resume: bool=False):
    for model_name in model_names:
        llm = TransformerLLM(model_name, dtype=torch.bfloat16)
        llm.merge_with_peft(get_checkpoint_path(model_name=model_name).as_posix())

        generator = ReWOOGenerate.init_transformer_llm(llm=llm)
        for dataset in datasets:
            generate_responses(
                dataset, 
                model_name=get_model_name_identifer(model_name=model_name, fine_tuned=True), 
                generator=generator, 
                save_dir=SAVE_DIR.as_posix(), 
                first_n=first_n,
                dataset_split=dataset_split,
                overwrite=not resume
            )
            dataset.clear_cache()

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
        generate_responses_for_gemini_models(datasets, GEMINI_MODELS, first_n=first_n, dataset_split="test")
        # generate_responses_for_local_models_before_fine_tuning(datasets, TRANSFORMER_MODELS, first_n=first_n, dataset_split="test")
        # generate_responses_for_local_models_after_fine_tuning(datasets, TRANSFORMER_MODELS, first_n=first_n, dataset_split="test")
        
        all_model_names = GEMINI_MODELS + [
                get_model_name_identifer(model_name, fine_tuned=False) for model_name in TRANSFORMER_MODELS
            ] + [
                get_model_name_identifer(model_name, fine_tuned=True) for model_name in TRANSFORMER_MODELS
            ]
        df = evaluate_all(all_model_names, datasets, save_dir=SAVE_DIR.as_posix(), use_transformated_answers=False)
        print(df)
    else:
        models = [args.model_name]
        generate_responses_for_local_models_before_fine_tuning(datasets, models, first_n=first_n, dataset_split="test")
        generate_responses_for_local_models_after_fine_tuning(datasets, models, first_n=first_n, dataset_split="test", resume=args.resume)
        
        all_model_names = GEMINI_MODELS + \
            [get_model_name_identifer(model_name, fine_tuned=False) for model_name in models] + \
            [get_model_name_identifer(model_name, fine_tuned=True) for model_name in models]
        df = evaluate_all(all_model_names, datasets, save_dir=SAVE_DIR.as_posix(), use_transformated_answers=False)
        print(df)
