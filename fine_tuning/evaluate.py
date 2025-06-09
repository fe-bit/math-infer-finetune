from math_datasets.generators import Generate, generate_responses, TransformersGenerate
from math_datasets.datasets import Dataset
from math_datasets.fine_tuning.llm.transformer_llm import TransformerLLM
from math_datasets.evaluator import evaluate_all


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
