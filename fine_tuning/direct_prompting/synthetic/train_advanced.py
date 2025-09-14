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
from datasets import Dataset as HFDataset
from tqdm import tqdm
load_dotenv()

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    force=True
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
    return parser.parse_args()

args = parse_args()
model_name = args.model_name
resume = args.resume

WORKING_DIR = Path(__file__).parent
output_dir = WORKING_DIR / f"training-output/{model_name}"

print(f"Using model: {model_name}")
print(f"Output directory: {output_dir}")
print(f"Resume from checkpoint: {resume}")

def get_dataset():
    p = WORKING_DIR / "train_data.jsonl"
    ds = load_dataset("json", data_files=p.as_posix(), split="train")
    
    # First split into train/test
    ds = ds.train_test_split(test_size=0.015, seed=42)
    return ds
    
    # Select balanced samples for test set and get remaining samples
    def select_samples_per_dataset_with_remainder(dataset, samples_per_dataset=25):
        """Select samples per dataset and return both selected and remaining samples."""
        selected_samples = []
        remaining_samples = []
        
        # Group by dataset
        dataset_groups = {}
        for sample in dataset:
            ds_name = sample["dataset"]
            if ds_name not in dataset_groups:
                dataset_groups[ds_name] = []
            dataset_groups[ds_name].append(sample)
        
        # Select samples_per_dataset from each group
        for ds_name, samples in dataset_groups.items():
            selected_count = min(samples_per_dataset, len(samples))
            selected_samples.extend(samples[:selected_count])
            remaining_samples.extend(samples[selected_count:])  # Add the rest
            print(f"Selected {selected_count} samples from {ds_name} for evaluation")
            print(f"Added {len(samples[selected_count:])} remaining {ds_name} samples to training")
        
        return selected_samples, remaining_samples
    
    # Get balanced test set and remaining samples
    test_samples, remaining_test_samples = select_samples_per_dataset_with_remainder(
        ds["test"], samples_per_dataset=25
    )
    
    # Create new datasets
    ds["test"] = HFDataset.from_list(test_samples)
    
    # Add remaining test samples back to training set
    original_train_list = [sample for sample in ds["train"]]
    combined_train_list = original_train_list + remaining_test_samples
    ds["train"] = HFDataset.from_list(combined_train_list)
    
    print(f"Final train samples: {len(ds['train'])} (original: {len(original_train_list)}, added: {len(remaining_test_samples)})")
    print(f"Final test samples: {len(ds['test'])}")
    
    return ds

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Ensure pad token is set

def format_chat(example):
    chat_messages = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_special_tokens=False
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
        model = trainer.model  # Access model from trainer
        model.eval()
        device = next(model.parameters()).device

        if model is None:
            raise ValueError("Model must be provided for generation-based evaluation")
        
        model.eval()
        device = next(model.parameters()).device
        
        # GSM8K evaluation
        correct = {}

        for sample in tqdm(dataset, desc="Math-Evaluation"):
            question = sample["question"]            
            inputs = tokenizer(question, return_tensors="pt", max_length=1024, truncation=True).to(device)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=1024)
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            pred_answer = Dataset.extract_answer(generated_text)
            true_answer = Dataset.extract_answer(sample["answer"])
            
            ds_name = sample["dataset"]
            if ds_name not in correct:
                correct[ds_name] = []
            correct[ds_name].append(is_correct_answer(pred_answer, true_answer))
            torch.cuda.empty_cache()  # Clear cache to avoid memory issues

        accuracies = {ds_name: float(np.mean(correct_list)) * 100 for ds_name, correct_list in correct.items()}      
        accuracies["avg_accuracy"] = float(np.mean(list(accuracies.values())))
        return accuracies
    
    return compute_metrics

class CustomSFTTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

training_args = SFTConfig(
    model_init_kwargs={
        "torch_dtype": precision_str,
        "device_map": "auto",
        "quantization_config": quantization_config,
    },
    output_dir=output_dir.as_posix(),
    num_train_epochs=2,
    learning_rate=2e-5,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    per_device_eval_batch_size=1,
    eval_accumulation_steps=1,
    gradient_checkpointing=True,
    save_strategy="steps",
    save_steps=200, # TODO: Adjust based on dataset size
    save_total_limit=3,
    logging_steps=25,
    eval_strategy="steps",
    eval_steps=200, # TODO: Adjust based on dataset size
    load_best_model_at_end=True,
    metric_for_best_model="eval_avg_accuracy",
    greater_is_better=True,
    dataset_text_field="text",
    max_grad_norm=2.0,
    weight_decay=0.01,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    max_seq_length=1024,
)

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj"],
    modules_to_save=["lm_head", "embed_tokens"],
    task_type="CAUSAL_LM",
    bias="lora_only",
    use_rslora=False,
)

trainer = CustomSFTTrainer(
    model=model_name,
    args=training_args,
    peft_config=peft_config,
    train_dataset=ds["train"],
    eval_dataset=ds["test"].select(range(45)),
    # callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    compute_metrics=get_compute_metrics(tokenizer, ds["test"].select(range(100))),
)

if resume:
    trainer.train(resume_from_checkpoint=True)
else:
    trainer.train()

trainer.save_model(output_dir.as_posix())