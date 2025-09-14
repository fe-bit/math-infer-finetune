import torch
import re
import numpy as np
from transformers import AutoTokenizer, EarlyStoppingCallback
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
from datasets import load_dataset
from pathlib import Path
import os
import logging
import sys
import argparse
from dotenv import load_dotenv
from math_datasets.datasets import Dataset
from tqdm import tqdm
from math_datasets.generators import TransformersGenerate
from math_datasets.fine_tuning.llm import TransformerLLM
import torch.nn.functional as F


load_dotenv()

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,  # or DEBUG
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    force=True  # Overwrites previous configs
)

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a language model with LoRA and SFTTrainer.")
    parser.add_argument(
        "model_name",
        type=str,
        help="The Hugging Face model name or path to fine-tune."
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the latest checkpoint.",
    )
    parser.add_argument(
        "--quantized",
        action="store_true",
        help="Use quantization for the model.",
    )
    parser.add_argument(
        "--use-verifier-loss",
        action="store_true",
        help="Use verifier loss during training.",
    )
    return parser.parse_args()

args = parse_args()
model_name = args.model_name
resume = args.resume

WORKING_DIR = Path(__file__).parent
output_dir = WORKING_DIR / f"training-output/{model_name}"

print(f"Using model: {model_name}")
print(f"Output directory: {output_dir}")
print(f"Resume from checkpoint: {resume}")
print(f"Use Verifier Loss: {args.use_verifier_loss}")

def get_dataset():
    p_train = WORKING_DIR / "train_data.jsonl"
    ds = load_dataset(
        "json", 
        data_files={
            "train": p_train.as_posix(),
        }
    )
    ds = ds["train"].train_test_split(test_size=0.1, seed=42)
    return ds

tokenizer = AutoTokenizer.from_pretrained(model_name)

def format_chat(example):
    messages = example["messages"]
    chat_messages = tokenizer.apply_chat_template(
        messages,
        tokenize=False, # Don't tokenize into IDs yet
        add_special_tokens=False # Apply model's specific start/end tokens, e.g., <s> and </s>
    )
    return {"text": chat_messages}

def format_test_example(example):
    """Format GSM8K/SVAMP questions as chat messages."""
    messages = [example["messages"][0]]
    chat_messages = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_special_tokens=False
    )
    return {"question": chat_messages, "answer": example["messages"][1]["content"]}


ds = get_dataset()
ds = ds.map(format_chat, batched=False)
ds = ds.map(format_test_example, batched=False)
print(ds)

if torch.cuda.is_available():
    device_map_strategy = "auto"
    if torch.cuda.is_bf16_supported():
        precision_str = "bfloat16"
    else:
        precision_str = "float16"
    from math_datasets.fine_tuning.llm.transformer_llm import TransformerLLM
    quantization_config = TransformerLLM.get_quantization_config() if args.quantized else None
elif torch.backends.mps.is_available():
    device_map_strategy = "auto"
    precision_str = "bfloat16"
    
    from math_datasets.fine_tuning.llm.transformer_llm import TransformerLLM
    quantization_config = TransformerLLM.get_quantization_config() if args.quantized else None
else:
    device_map_strategy = "cpu"
    precision_str = "float32"
    quantization_config = None
    if args.quantized:
        print("Warning: Quantization is not supported on CPU. Using without quantization.")


def get_compute_metrics(tokenizer, dataset):
    def is_correct_answer(prediction: float|None, label: float|None) -> bool:
        if prediction is None:
            return False
        elif label is None:
            raise ValueError("Label cannot be None")
        return abs(prediction - label) < 1e-5

    def compute_metrics(eval_pred):
        model = trainer.model
        if model is None:
            raise ValueError("Model must be provided for generation-based evaluation")
        
        model.eval()
        llm = TransformerLLM.using(
            model_name=model_name,
            model=model,
            tokenizer=tokenizer
        )
        
        generator = TransformersGenerate(llm)
        correct = []
        for sample in tqdm(dataset, desc="Evaluating DP"):
            try:
                entry = {}
                resp = generator.generate(prompt=sample["question"], entry=entry)
                float_answer = Dataset.extract_answer(resp)
                
                is_correct = is_correct_answer(float_answer, Dataset.extract_answer(sample["answer"]))
            except Exception as e:
                is_correct = False

            correct.append(is_correct)

        return {
            "avg_accuracy": np.mean(correct) * 100,
        }
    
    return compute_metrics

training_args = SFTConfig(
    model_init_kwargs={
        "torch_dtype": precision_str,
        "device_map": "auto",
        "quantization_config": quantization_config,
    },
    output_dir=output_dir.as_posix(),
    num_train_epochs=3,
    learning_rate=2e-5,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    per_device_eval_batch_size=1,
    eval_accumulation_steps=1,
    gradient_checkpointing=True,
    save_strategy="steps",
    save_steps=200, # TODO: Adjust based on dataset size
    save_total_limit=2,
    logging_steps=50,
    eval_strategy="steps",
    eval_steps=200, # TODO: Adjust based on dataset size
    load_best_model_at_end=True,
    metric_for_best_model="eval_avg_accuracy",
    greater_is_better=True,
    dataset_text_field="text",
    weight_decay=0.01,
    warmup_ratio=0.1,
    max_seq_length=1024,
)

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules="all-linear",
    # modules_to_save=["lm_head", "embed_tokens"],
    task_type="CAUSAL_LM",
    bias="lora_only",
    use_rslora=True, # Use RSLORA for better performance
)

print("Using standard SFTTrainer without verifier loss.")
trainer = SFTTrainer(
    model=model_name,
    args=training_args,
    peft_config=peft_config,
    train_dataset=ds["train"],
    eval_dataset=ds["test"].select(range(45)),
    # callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    compute_metrics=get_compute_metrics(tokenizer, ds["test"].select(range(45))),
)

if resume:
    trainer.train(resume_from_checkpoint=True)
else:
    trainer.train()

trainer.save_model(output_dir.as_posix())