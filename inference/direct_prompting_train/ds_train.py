import pandas as pd
from pathlib import Path


SAVE_DIR = Path(__file__).parent.as_posix()

from math_datasets.datasets import Dataset, GSM8K, SVAMP
from math_datasets.evaluator import evaluate_all
from math_datasets.training_data import get_training_data
import os
from dotenv import load_dotenv
from pathlib import Path
from transformers import AutoTokenizer

load_dotenv(override=True)

SAVE_DIR = Path(__file__).parent.as_posix()

GEMINI_MODELS = [
    "gemini-2.0-flash",
]

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

def format_chat(example):
    user_message = example["input"]
    assistant_message = example["output"]
    messages = [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": assistant_message}
    ]
    return messages
    chat_messages = tokenizer.apply_chat_template(
        messages,
        tokenize=False, # Don't tokenize into IDs yet
        add_special_tokens=False # Apply model's specific start/end tokens, e.g., <s> and </s>
    )
    return chat_messages

if __name__ == "__main__":
    datasets = [GSM8K, SVAMP]
    
    df_train1 = get_training_data(GEMINI_MODELS[0], datasets[0], SAVE_DIR)
    df_train2 = get_training_data(GEMINI_MODELS[0], datasets[1], SAVE_DIR)

    df_train = pd.concat([df_train1, df_train2])
    # shuffle the training data
    df_train = df_train.sample(frac=1, random_state=42).reset_index(drop=True)
    df_train["messages"] = df_train.apply(format_chat, axis=1)
    df_train = df_train[["messages"]]
    print(df_train.head())
    df_train.to_json(Path(SAVE_DIR) / "train_data.jsonl", orient="records", lines=True)
    df_train.to_excel(Path(SAVE_DIR) / "train_data.csv", index=False)
    print(len(df_train), "training examples generated.")
