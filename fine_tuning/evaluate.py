import os
from math_datasets.generators import Generate, generate_responses
from math_datasets.datasets import Dataset
from math_datasets.fine_tuning.llm.utils import get_latest_checkpoint_dir
from math_datasets.fine_tuning.llm import LLM
from math_datasets.fine_tuning.llm.transformer_llm import TransformerLLM
from math_datasets.evaluator import evaluate_all


class LocalGenerator(Generate):
    def __init__(self, model: LLM, merge_with_peft_dir: str = None):
        self.model = model
        if merge_with_peft_dir is not None:
            checkpoint_path = get_latest_checkpoint_dir(merge_with_peft_dir)
            self.model.merge_with_peft(checkpoint_path)

    def generate(self, prompt:str,  entry: dict[str, str]={}) -> str:
        return self.model.generate(
            prompt, 
            max_new_tokens=512,
            temperature=None,
            top_p=None,
            top_k=None
        )
    
def evaluate_one(generator: Generate, datasets: list[Dataset], model_name: str, first_n: int=100):
    for dataset in datasets:
        generate_responses(
            dataset, 
            model_name=model_name, 
            generator=generator, 
            save_dir=os.getcwd(), 
            first_n=first_n
        )

    
def evaluate(model_name: str, output_dir: str, datasets: list[Dataset], first_n: int=100, with_peft: bool=False):
    eval_name_before_training = model_name + "_before_training"
    eval_name_after_training = model_name + "_after_training"

    evaluate_one(generator=LocalGenerator(
        TransformerLLM(model_name),
    ), datasets=datasets, first_n=first_n, model_name=eval_name_before_training)

    if with_peft:
        evaluate_one(
            generator=LocalGenerator(
                model=TransformerLLM(model_name=model_name), 
                merge_with_peft_dir=get_latest_checkpoint_dir(output_dir)
            ),
            datasets=datasets, 
            first_n=first_n, 
            model_name=eval_name_before_training
        )
    else:
        evaluate_one(generator=LocalGenerator(
            TransformerLLM.from_trained(model_name=model_name, checkpoint_path=get_latest_checkpoint_dir(output_dir)),
        ), datasets=datasets, first_n=first_n, model_name=eval_name_before_training)

    ### Evaluation ###
    df = evaluate_all([eval_name_before_training, eval_name_after_training], datasets=datasets, save_dir=output_dir, use_transformated_answers=False)
    print(df)