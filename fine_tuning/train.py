from transformers import TrainingArguments, AutoTokenizer, AutoModelForCausalLM, EarlyStoppingCallback
import os
import logging
import sys
from pathlib import Path
from peft import LoraConfig
from datasets import Dataset as HFDataset
import pandas as pd
import torch
from trl import SFTTrainer, SFTConfig
import argparse
from dotenv import load_dotenv


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
    # Add more arguments here as needed (batch size, epochs, etc.)
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
    p = WORKING_DIR / "train_data.xlsx"
    df = pd.read_excel(p)
    df = df.dropna(axis=0)
    ds = HFDataset.from_pandas(df)
    ds = ds.train_test_split(test_size=0.1, seed=42)
    return ds

ds = get_dataset()

def format_dataset(ds):
    def format_chat(example):
        user_message = example["input"]
        assistant_message = example["output"]
        messages = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message}
        ]
        return {
            "messages": messages
        }
    ds = ds.map(format_chat, batched=False)
    return ds

ds = format_dataset(ds)

training_args = TrainingArguments(
    output_dir=output_dir.as_posix(),
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    
    num_train_epochs=8, # we use EarlyStoppingCallback to stop training if eval_loss doesn't improve for 3 evals

    logging_steps=25,

    save_strategy="steps",
    save_steps=100,

    save_total_limit=2,
    eval_strategy="steps",
    eval_steps=100,

    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    logging_dir="./logs",
    report_to="none",
)


lora_config = LoraConfig(
    r=4,
    lora_alpha=8,
    lora_dropout=0.05,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj", 
        "gate_proj", "up_proj", "down_proj"
    ],
    task_type="CAUSAL_LM",
    bias="lora_only",
    modules_to_save=["lm_head", "embed_token"],
)

if torch.cuda.is_available():
    print("CUDA is available! Using GPU.")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    device_map_strategy = "cuda"
elif torch.backends.mps.is_available():
    print("MPS (Metal Performance Shaders) is available! Using Apple Silicon GPU.")
    device_map_strategy = "mps"
else:
    print("CUDA is not available. Using CPU.")
    num_threads = int(os.environ.get("SLURM_CPUS_PER_TASK", 1)) # Default to 1 if not in Slurm
    torch.set_num_threads(num_threads)
    print(f"PyTorch using {torch.get_num_threads()} CPU threads.")
    device_map_strategy = "cpu"


training_args = SFTConfig(packing=True, **training_args.to_dict())
os.makedirs(output_dir, exist_ok=True)
with open(f"{output_dir}/training_args.json", "w") as f:
    f.write(training_args.to_json_string())

trainer = SFTTrainer(
    model=model_name,
    args=training_args,
    peft_config=lora_config, # is None if not using LoRA
    train_dataset=ds["train"],
    eval_dataset=ds["test"],
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

if resume:
    trainer.train(resume_from_checkpoint=True)
else:
    trainer.train()
    
trainer.save_model(output_dir.as_posix())