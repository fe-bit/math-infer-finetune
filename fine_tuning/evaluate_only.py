import os
from math_datasets.evaluator import evaluate_all
from math_datasets import GSM8K


if __name__ == "__main__":
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    eval_name_before_training = model_name + "_before_training"
    eval_name_after_training = model_name + "_after_training"
    output_dir = f"./fine-tune-dp/GSM8K/{model_name}"

    ### Evaluation ###
    df = evaluate_all([eval_name_before_training, eval_name_after_training], datasets=[GSM8K], save_dir=os.getcwd(), use_transformated_answers=False)
    print(df)