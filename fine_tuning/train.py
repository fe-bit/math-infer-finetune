from transformers import TrainingArguments, AutoTokenizer, AutoModelForCausalLM, EarlyStoppingCallback
import os
import logging
import sys
from pathlib import Path
from peft import LoraConfig
from datasets import Dataset as HFDataset, load_dataset
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

# Add at the top of your train.py, before model loading
if 'LOCAL_RANK' in os.environ:
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
else:
    device = torch.device('cuda:0')


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
        "--trainingeval",
        action="store_true",
        help="Eval during training.",
    )
    # Add more arguments here as needed (batch size, epochs, etc.)
    return parser.parse_args()

args = parse_args()
model_name = args.model_name
resume = args.resume
training_eval = args.trainingeval

WORKING_DIR = Path(__file__).parent
output_dir = WORKING_DIR / f"training-output/{model_name}"

print(f"Using model: {model_name}")
print(f"Output directory: {output_dir}")
print(f"Resume from checkpoint: {resume}")

def get_dataset():
    # return load_dataset("openai/gsm8k", "main")
    p = WORKING_DIR / "train_data.jsonl"
    ds = load_dataset("json", data_files=p.as_posix(), split="train")
    ds = ds.train_test_split(test_size=0.1, seed=42)
    return ds

ds = get_dataset()

def format_dataset(ds):
    def format_chat(example):
        user_message = example["question"] #example["input"]
        assistant_message = example["answer"] #example["output"]
        messages = [
            {"role": "system", "content": "You are a math professor."},
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message}
        ]
        return {
            "messages": messages
        }
    ds = ds.map(format_chat, batched=False)
    return ds

# ds = format_dataset(ds)


if torch.cuda.is_available():
    print("CUDA is available! Using GPU.")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    device_map_strategy = "cuda"
    precision_str = "float16"
elif torch.backends.mps.is_available():
    print("MPS (Metal Performance Shaders) is available! Using Apple Silicon GPU.")
    device_map_strategy = "mps"
    precision_str = "bfloat16"
else:
    print("CUDA is not available. Using CPU.")
    num_threads = int(os.environ.get("SLURM_CPUS_PER_TASK", 1)) # Default to 1 if not in Slurm
    torch.set_num_threads(num_threads)
    print(f"PyTorch using {torch.get_num_threads()} CPU threads.")
    device_map_strategy = "cpu"
    precision_str = "float32"

training_args = SFTConfig(
    model_init_kwargs={
        "torch_dtype": precision_str,
        "device_map": device_map_strategy,
    },
    output_dir=output_dir.as_posix(),
    num_train_epochs=3,
    per_device_train_batch_size=1,       
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,                  
    packing=False,
    logging_steps=100,    
    
    save_strategy="steps",
    save_steps=200,
    eval_strategy="no",
)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules="all-linear",
    modules_to_save=["lm_head", "embed_token"],
    task_type="CAUSAL_LM",
)

trainer = SFTTrainer(
    model=model_name,
    args=training_args,
    peft_config=peft_config,
    train_dataset=ds["train"],
)

if resume:
    trainer.train(resume_from_checkpoint=True)
else:
    trainer.train()
    
trainer.save_model(output_dir.as_posix())