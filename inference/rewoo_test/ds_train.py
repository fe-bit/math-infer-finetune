import pandas as pd
from pathlib import Path
from math_datasets.datasets import Dataset, GSM8K, SVAMP
from math_datasets.evaluator import evaluate_detail
from math_datasets.training_data import get_training_data
from dotenv import load_dotenv
from math_datasets.generators.rewoo import ReWOOModel
load_dotenv(override=True)

SAVE_DIR = Path(__file__).parent.as_posix()

GEMINI_MODELS = [
    # "gemma-3-27b-it",
]

prompt = ReWOOModel.get_prompt(with_examples=False)
print(prompt)

def format_chat(example):
    user_message = example["input"]
    assistant_message = example["output"]
    messages = [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": assistant_message}
    ]
    return messages

if __name__ == "__main__":
    datasets = [GSM8K, SVAMP]

    # Evaluate all datasets
    df_gsm8k = evaluate_detail("gemma-3-27b-it", GSM8K, SAVE_DIR, use_transformated_answers=False)
    df_svamp = evaluate_detail("gemma-3-27b-it", SVAMP, SAVE_DIR, use_transformated_answers=False)
    
    # select 50 first entries in both dataframes for now
    df_gsm8k = df_gsm8k.iloc[:50]
    df_svamp = df_svamp.iloc[:50]

    df_gsm8k["dataset"] = datasets[0].name
    df_svamp["dataset"] = datasets[1].name
    df = pd.concat([df_gsm8k, df_svamp], ignore_index=True)
    df = df[df["is_correct"] == True].copy()
    # shuffle the training data
    df.loc[:, "sft_label"] = df["model_history"].apply(
        lambda x: x[0]["plan"]["plan_string"]
    )
    df = df[["question", "sft_label", "dataset"]]
    df = df.rename(columns={"sft_label": "output"})
    df["input"] = df["question"].apply(
        lambda x: prompt.format(task=x)
    )
    df = df[["question", "input", "output", "dataset"]]
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)


    df["messages"] = df.apply(format_chat, axis=1)
    df = df[["messages", "dataset"]]
    p = Path(__file__).parent / "rewoo_test_data.jsonl"
    df.to_json(p.as_posix(), orient="records", lines=True)
