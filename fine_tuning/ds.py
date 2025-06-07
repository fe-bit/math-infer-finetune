from math_datasets.datasets import Dataset, GSM8K
from datasets import load_dataset
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
        p = Path(__file__).parent / "train_data.csv"
        df = pd.read_csv(p)
        df = df.dropna(axis=0)
        ds = HFDataset.from_pandas(df)
        
        # ds = load_dataset("csv", data_files=p.as_posix())["train"]
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

    # @classmethod
    # def load_and_tokenize_dataset(cls, tokenizer, max_length=10000):
    #     def format_and_tokenize(example):
    #         user_message = cls.get_input_text(example)
    #         assistant_message = cls.get_output_text(example)
    #         messages = [
    #             {"role": "user", "content": user_message},
    #             {"role": "assistant", "content": assistant_message}
    #         ]
            
    #         # Just format the text - no manual label creation needed
    #         formatted_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    #         full = tokenizer(formatted_text, truncation=True, max_length=max_length, padding=True)
    #         return full
    #         input_ids = full["input_ids"]
    #         user_prompt = tokenizer.apply_chat_template(
    #             [{"role": "user", "content": user_message}],
    #             tokenize=False,
    #             add_generation_prompt=True  # appends assistant prompt
    #         )
    #         user_tokens = tokenizer(user_prompt, truncation=False, padding=False)

    #         prompt_len_after_truncation = len(user_tokens["input_ids"])
    #         actual_mask_len = min(prompt_len_after_truncation, len(input_ids))
    #         labels = [-100] * len(input_ids)
    #         if actual_mask_len < len(input_ids): # Check if there is an assistant part to unmask
    #             for i in range(actual_mask_len, len(input_ids)):
    #                 labels[i] = input_ids[i]

    #         # Ensure labels match input_ids length exactly
    #         if len(labels) > len(input_ids):
    #             labels = labels[:len(input_ids)]
    #         elif len(labels) < len(input_ids):
    #             labels.extend([-100] * (len(input_ids) - len(labels)))

    #         assert len(labels) == len(input_ids), f"Mismatch in lengths: len(labels)={len(labels)}, len(input_ids)={len(input_ids)}. Full example: {example}"
    #         assert len(full["attention_mask"]) == len(input_ids), f"Mismatch in lengths: len(attention_mask)={len(full["attention_mask"])}, len(input_ids)={len(input_ids)}. Full example: {example}"

    #         return {
    #             "input_ids": input_ids,
    #             "labels": labels,
    #             "attention_mask": full["attention_mask"]
    #         }

    #     tokenized_dataset = cls.get_dataset().map(format_and_tokenize, batched=False)
    #     tokenized_dataset.set_format("torch")
    #     return tokenized_dataset
    
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
                "prompt": tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False
                )
            }

        def tokenize(example):
            return tokenizer(example["prompt"], truncation=True)

        ds = cls.get_dataset()
        ds = ds.map(format_chat, batched=False)
        tokenized_ds = ds.map(tokenize, batched=False)
        return tokenized_ds
