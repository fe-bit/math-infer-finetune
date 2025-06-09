from train_utils import train

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
train(model_name, resume=True)
