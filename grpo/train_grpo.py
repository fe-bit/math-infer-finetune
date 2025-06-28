from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
from math_datasets.datasets import GSM8K
import logging
import sys
from pathlib import Path

WORKING_DIR = Path(__file__).parent

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,  # or DEBUG
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    force=True  # Overwrites previous configs
)


dataset = load_dataset("openai/gsm8k", "main")

def add_prompt(example):
    """Format GSM8K/SVAMP questions as chat messages."""
    return {
        "prompt": GSM8K.get_input_text(example),
        "completion": GSM8K.get_output_text(example),
    }

dataset = dataset.map(add_prompt, batched=False)

# Define the reward function, which rewards completions that are close to 20 characters
def reward_len(completions, **kwargs):
    return [-abs(20 - len(completion)) for completion in completions]

output_dir = WORKING_DIR / "training-output" / "HuggingFaceTB/SmolLM2-135M-Instruct"

training_args = GRPOConfig(
    output_dir=output_dir.as_posix(),
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=1,

    per_device_eval_batch_size=1,
    eval_accumulation_steps=1
)
trainer = GRPOTrainer(
    model="HuggingFaceTB/SmolLM2-135M-Instruct",
    reward_funcs=reward_len,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"].select(range(45))
)
trainer.train()
trainer.save_model(output_dir.as_posix())