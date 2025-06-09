from math_datasets.datasets import Dataset, GSM8K
from pathlib import Path
import pandas as pd
from datasets import Dataset as HFDataset


class TrainingGSM8KPlannerDataset(Dataset):
    name="GSM8K-Gemini-2.0-Flash-Training"
    gsm8k_ds = GSM8K

    @classmethod
    def get_input_text(cls, example):
        return example["input"]

    @classmethod
    def get_output_text(cls, example):  
        return example["output"]

    @classmethod
    def get_dataset(cls):
        p = Path(__file__).parent / "train_data.xlsx"
        df = pd.read_excel(p)
        df = df.dropna(axis=0)
        ds = HFDataset.from_pandas(df)
        ds = ds.train_test_split(test_size=0.1, seed=42)
        return ds
    
    @classmethod
    def get_float_answer(cls, example):
        raise NotImplementedError("This dataset does not provide float answers.")
    
    @classmethod
    def is_answer_correct(cls, entry, use_transformated_answers:bool=True) -> bool:
        raise NotImplementedError("This dataset does not support answer correctness checking.")

    @classmethod
    def format_input_evaluate(cls, example, prompt_prefix=""):
        return (
            "<|im_start|>user\n" + prompt_prefix + cls.get_input_text(example) + "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
    
    @classmethod
    def load_and_tokenize_dataset(cls, tokenizer, max_length=10000):
        def format_chat(example):
            user_message = cls.get_input_text(example)
            assistant_message = cls.get_output_text(example)
            messages = [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": assistant_message}
            ]
            return {
                "messages": messages
            }

        ds = cls.get_dataset()
        ds = ds.map(format_chat, batched=False)
        return ds
