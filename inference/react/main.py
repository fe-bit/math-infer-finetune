from math_datasets.datasets import Dataset, GSM8K, SVAMP
from math_datasets.generators import generate_responses
from math_datasets.evaluator import evaluate_all, evaluate_detail
from dotenv import load_dotenv
from typing import List, Literal
from pathlib import Path
from math_datasets.generators import generate_responses
import argparse
from math_datasets.generators.react import ReactGenerate
from langchain_ollama import ChatOllama



load_dotenv(override=True)

SAVE_DIR = Path(__file__).parent
RESULTS_DIR = Path(__file__).parent.parent.parent / "results" / "react"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OLLAMA_MODELS = [
    "qwen2.5:0.5b",
    "qwen3:0.6b",
    "llama3.2:1b",
    "qwen3:1.7b",
    "smollm2:1.7b",
]
    
def get_ollama_model_name_identifer(model_name: str) -> str:
    return "ollama/" + model_name.replace(":", "_")

def generate_responses_for_ollama_models(datasets: List[Dataset], model_names: List[str], first_n: int|None=None, dataset_split: Literal["test", "train"]="test"):
    for model_name in model_names:
        model = ChatOllama(model=model_name, temperature=0, num_predict=2024)
        generator = ReactGenerate(model=model, use_4_tools=False)

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
            df = evaluate_detail(get_ollama_model_name_identifer(model_name), dataset=dataset, use_transformated_answers=False, additional_metrics=ReactGenerate, save_dir=SAVE_DIR.as_posix(), use_first_n=first_n)
            f = RESULTS_DIR / dataset.name / f"{get_ollama_model_name_identifer(model_name)}.csv"
            f.parent.mkdir(parents=True, exist_ok=True)
            if not df.empty:
                df.to_csv(f, index=False)

def main(first_n: int|None=None):
    datasets = [SVAMP, GSM8K]
    generate_responses_for_ollama_models(datasets, OLLAMA_MODELS, first_n=first_n, dataset_split="test")
    all_model_names = [get_ollama_model_name_identifer(model_name) for model_name in OLLAMA_MODELS]
    df = evaluate_all(all_model_names, datasets, save_dir=SAVE_DIR.as_posix(), use_transformated_answers=False, use_first_n=first_n)
    df.to_csv(SAVE_DIR / "react_evaluation.csv", index=False)
    print(df)

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
        generate_responses_for_ollama_models(datasets, OLLAMA_MODELS, first_n=first_n, dataset_split="test")
        all_model_names = [get_ollama_model_name_identifer(model_name) for model_name in OLLAMA_MODELS]
        df = evaluate_all(all_model_names, datasets, save_dir=SAVE_DIR.as_posix(), use_transformated_answers=False, use_first_n=first_n)
        df.to_csv(SAVE_DIR / "react_evaluation.csv", index=False)
        print(df)
    else:
        models = [args.model_name]
        generate_responses_for_ollama_models(datasets, OLLAMA_MODELS, first_n=first_n, dataset_split="test")        
        all_model_names = [get_ollama_model_name_identifer(model_name) for model_name in OLLAMA_MODELS]
        df = evaluate_all(all_model_names, datasets, save_dir=SAVE_DIR.as_posix(), use_transformated_answers=False)
        print(df)
