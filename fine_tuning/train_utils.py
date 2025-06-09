


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


def train(model_name: str, resume: bool = True):
    output_dir = SAVE_DIR / f"training-output/{model_name}"

    training_args = TrainingArguments(
        output_dir=output_dir.as_posix(),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        
        num_train_epochs=5, # we use EarlyStoppingCallback to stop training if eval_loss doesn't improve for 3 evals

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

    print("Starting Training:", model_name)
    ### Training ###
    model = TransformerLLM(model_name)
    trainer = CustomTrainer()
    trainer.train(
        llm=model, 
        output_dir=output_dir.as_posix(), 
        dataset=TrainingGSM8KPlannerDataset, 
        resume_from_checkpoint=resume, 
        lora_config=lora_config,
        training_args=training_args
    )
    print(output_dir, SAVE_DIR)
    datasets = [SVAMP, GSM8K]
    evaluate(model_name=model_name, output_dir=SAVE_DIR.as_posix(), datasets=datasets, first_n=50, checkpoint_dir=output_dir.as_posix(), with_peft=True)
