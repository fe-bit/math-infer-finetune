import pandas as pd
from pathlib import Path
from math_datasets.datasets import Dataset, GSM8K, SVAMP
from math_datasets.evaluator import evaluate_all
from math_datasets.training_data import get_training_data
from dotenv import load_dotenv

load_dotenv(override=True)

SAVE_DIR = Path(__file__).parent.as_posix()

GEMINI_MODELS = [
    "gemma-3-27b-it",
]

if __name__ == "__main__":
    datasets = [GSM8K, SVAMP]
    first_n = None

    print(len(GSM8K.get_dataset()["train"]))
    print(len(SVAMP.get_dataset()["train"]))

    # Evaluate all datasets
    df = evaluate_all(GEMINI_MODELS, datasets, save_dir=SAVE_DIR, use_transformated_answers=False)
    print(df)
    
    df_train1 = get_training_data(GEMINI_MODELS[0], datasets[0], SAVE_DIR)
    df_train2 = get_training_data(GEMINI_MODELS[0], datasets[1], SAVE_DIR)

    df_train = pd.concat([df_train1, df_train2])
    # shuffle the training data
    df_train = df_train.sample(frac=1, random_state=42).reset_index(drop=True)
    df_train.to_excel(Path(SAVE_DIR) / "train_data.xlsx", index=False)
    print(len(df_train), "training examples generated.")
