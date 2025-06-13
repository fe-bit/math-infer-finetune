from math_datasets.generators import Generate, generate_responses, TransformersGenerate
from math_datasets.datasets import Dataset, GSM8K, SVAMP
from math_datasets.fine_tuning.llm.transformer_llm import TransformerLLM
from math_datasets.evaluator import evaluate_all
from pathlib import Path
import argparse

def evaluate_one(generator: Generate, datasets: list[Dataset], model_name: str, save_dir: str, first_n: int=100, overwrite: bool=False):
    for dataset in datasets:
        generate_responses(
            dataset, 
            model_name=model_name, 
            generator=generator, 
            save_dir=save_dir, 
            first_n=first_n,
            overwrite=overwrite
        )

    
def evaluate(model_name: str, output_dir: str, datasets: list[Dataset], checkpoint_dir: str, first_n: int=100, with_peft: bool=True):
    eval_name_before_training = model_name + "/before_training"
    eval_name_after_training = model_name + "/after_training"

    evaluate_one(
        generator=TransformersGenerate(TransformerLLM(model_name)), 
        datasets=datasets, 
        first_n=first_n, 
        model_name=eval_name_before_training, 
        save_dir=output_dir,
        overwrite=False
    )

    if with_peft:
        llm = TransformerLLM(model_name=model_name)
        llm.merge_with_peft(checkpoint_dir)
        evaluate_one(
            generator=TransformersGenerate(model=llm),
            datasets=datasets, 
            first_n=first_n, 
            model_name=eval_name_after_training,
            save_dir=output_dir,
            overwrite=True
        )
        del llm
    else:
        llm = TransformerLLM.from_trained(model_name=model_name, checkpoint_path=checkpoint_dir)
        evaluate_one(
            generator=TransformersGenerate(model=llm), 
            datasets=datasets, 
            first_n=first_n, 
            model_name=eval_name_after_training, 
            save_dir=output_dir,
            overwrite=True
        )
        del llm

    ### Evaluation ###
    df = evaluate_all([eval_name_before_training, eval_name_after_training], datasets=datasets, save_dir=output_dir, use_transformated_answers=False, use_first_n=first_n)
    print(df)


if __name__:
    parser = argparse.ArgumentParser(description="Evaluate model performance before and after training")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", 
                       help="Name of the model to evaluate (default: Qwen/Qwen2.5-0.5B-Instruct)")
    parser.add_argument("--first-n", type=int, default=50,
                       help="Number of samples to evaluate (default: 50)")
    parser.add_argument("--with-peft", action="store_true", default=False,
                       help="Whether to use PEFT for fine-tuning")
    args = parser.parse_args()

    SAVE_DIR = Path(__file__).parent
    model_name = args.model_name
    output_dir = SAVE_DIR / f"training-output/{model_name}"

    print("Using model:", model_name)
    print("With Peft:", args.with_peft)
    datasets = [SVAMP, GSM8K]
    evaluate(model_name=model_name, output_dir=SAVE_DIR.as_posix(), datasets=datasets, first_n=args.first_n, checkpoint_dir=output_dir.as_posix(), with_peft=args.with_peft)
