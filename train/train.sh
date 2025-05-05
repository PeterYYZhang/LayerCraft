# Specify the config file path and the GPU devices to use
# export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=0

# Specify the config file path
export XFL_CONFIG=./train/config/config.yaml

# Specify the WANDB API key
# export WANDB_API_KEY='YOUR_WANDB_API_KEY'

echo $XFL_CONFIG
export TOKENIZERS_PARALLELISM=true

accelerate launch --main_process_port 13366 -m src.train.train