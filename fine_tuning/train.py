from train_utils import train

model_name = "HuggingFaceTB/SmolLM2-360M-Instruct"
train(model_name, resume=False)
