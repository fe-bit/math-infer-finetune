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
    return load_dataset("openai/gsm8k", "main")
    p = WORKING_DIR / "train_data.xlsx"
    df = pd.read_excel(p)
    df = df.dropna(axis=0)
    ds = HFDataset.from_pandas(df)
    ds = ds.train_test_split(test_size=0.1, seed=42)
    return ds

ds = get_dataset()

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Set pad token to eos token if not already set

def format_dataset(ds):
    def format_chat(example):
        user_message = example["question"] #example["input"]
        assistant_message = example["answer"] #example["output"]
        messages = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message}
        ]
        m = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            return_tensors=None
        )
        return {
            "text": m
        }
    ds = ds.map(format_chat, batched=False)
    return ds

ds = format_dataset(ds)


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

# Add these memory-saving parameters to your training configuration
training_args = SFTConfig(
    output_dir=output_dir.as_posix(),
    num_train_epochs=8,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,         
    gradient_accumulation_steps=16,
    gradient_checkpointing=True,          
    dataloader_pin_memory=False,          
    max_seq_length=512,
    packing=True,
    logging_steps=10,
    save_steps=500,
    eval_steps=500,
    warmup_steps=10,
    learning_rate=5e-5,
    fp16=True,                           
    remove_unused_columns=False,
    dataloader_num_workers=0,             
    eval_strategy="no",
    save_strategy="no",
    dataset_text_field="text"
)

lora_config = LoraConfig(
    r=4,                    # Reduce from 16 to 8
    lora_alpha=8,          # Reduce accordingly
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj", 
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    modules_to_save=["lm_head", "embed_token"],
)

trainer = SFTTrainer(
    model=model_name,
    args=training_args,
    peft_config=lora_config,
    train_dataset=ds["train"],
)

if resume:
    trainer.train(resume_from_checkpoint=True)
else:
    trainer.train()
    
trainer.save_model(output_dir.as_posix())