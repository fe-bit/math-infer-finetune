from math_datasets.fine_tuning.trainer import CustomTrainer
from math_datasets.fine_tuning.llm import TransformerLLM
from ds import TrainingGSM8KPlannerDataset
from transformers import TrainingArguments
from math_datasets.datasets import SVAMP, GSM8K
from evaluate import evaluate
import os
import logging
import sys
from pathlib import Path
from peft import LoraConfig


logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,  # or DEBUG
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    force=True  # Overwrites previous configs
)

os.environ["TMPDIR"] = os.path.expanduser("~/tmp")

SAVE_DIR = Path(__file__).parent

# model_name = "Qwen/Qwen2.5-1.5B-Instruct" # "Qwen/Qwen2.5-0.5B-Instruct"
model_name = "HuggingFaceTB/SmolLM2-360M-Instruct"
# model_name = "Qwen/Qwen2.5-0.5B-Instruct"

eval_name_before_training = model_name + "_before_training"
eval_name_after_training = model_name + "_after_training"
output_dir = SAVE_DIR / f"training-output/{model_name}"


training_args = TrainingArguments(
    output_dir=output_dir.as_posix(),
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=15,
    learning_rate=5e-5,
    warmup_steps=20,
    weight_decay=0.01,
    max_grad_norm=1.0,
    lr_scheduler_type="cosine",
    logging_steps=25,

    save_strategy="epoch",
    save_total_limit=2,                # <-- Keep best + most recent
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    eval_strategy="epoch",
    logging_dir="./logs",
    report_to="none",
)

model = TransformerLLM(model_name)
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.01,
    target_modules=[
        "q_proj", "k_proj", "v_proj", # "o_proj", 
        "gate_proj", "up_proj", "down_proj"
    ],
    task_type="CAUSAL_LM",
    bias="lora_only"
)

print("Starting Training:", model_name)
### Training ###
trainer = CustomTrainer()
trainer.train(
    llm=model, 
    output_dir=output_dir.as_posix(), 
    dataset=TrainingGSM8KPlannerDataset, 
    resume_from_checkpoint=True, 
    lora_config=lora_config,
    training_args=training_args
)

evaluate(model_name=model_name, output_dir=output_dir.as_posix(), datasets=[SVAMP, GSM8K], first_n=100, with_peft=True)
