"""
Dual-Condition Model Training Script

This script handles the training process for the OminiModel/OINModel dual-condition
Flux diffusion model. It sets up the training environment, loads and prepares datasets,
initializes the model with appropriate LoRA adapters, and manages the training process.

The script supports:
- Distributed training with proper rank detection
- Configuration via YAML files
- WandB logging integration
- Checkpoint saving and resuming
- Dataset filtering and preprocessing
"""

import os
import time
import yaml
import ast
import torch
import lightning as L
from typing import Dict, Any, Optional, List, Tuple
from torch.utils.data import DataLoader
from datasets import load_dataset, load_from_disk

from .data import LayerSubjectDataset
from .OIN import OINModel
from .callbacks import TrainingCallback


def get_rank() -> int:
    """
    Get the current process rank for distributed training.
    
    Returns:
        Process rank (0 for main process, >0 for other processes)
    """
    try:
        rank = int(os.environ.get("LOCAL_RANK", "0"))
    except:
        rank = 0
    return rank


def get_config() -> Dict[str, Any]:
    """
    Load configuration from the environment-specified YAML file.
    
    Returns:
        Dictionary containing configuration parameters
        
    Raises:
        AssertionError: If XFL_CONFIG environment variable is not set
    """
    config_path = os.environ.get("XFL_CONFIG")
    assert config_path is not None, "Please set the XFL_CONFIG environment variable"
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    return config


def init_wandb(wandb_config: Dict[str, Any], run_name: str) -> None:
    """
    Initialize Weights & Biases logging if configured.
    
    Args:
        wandb_config: WandB configuration parameters
        run_name: Name for this training run
        
    Note:
        Silently fails if WandB initialization fails, with error printed
    """
    import wandb

    try:
        assert os.environ.get("WANDB_API_KEY") is not None, "WANDB_API_KEY not set"
        wandb.init(
            project=wandb_config["project"],
            name=run_name,
            config={},
        )
    except Exception as e:
        print(f"Failed to initialize WandB: {e}")


def main() -> None:
    """
    Main training function.
    
    This function:
    1. Sets up the training environment
    2. Loads and prepares the dataset
    3. Initializes the model
    4. Configures the trainer
    5. Starts the training process
    """
    # Initialize training environment
    is_main_process, rank = get_rank() == 0, get_rank()
    torch.cuda.set_device(rank)
    config = get_config()
    training_config = config["train"]
    run_name = time.strftime("%Y%m%d-%H%M%S")
    
    # Set up output directories
    save_path = training_config.get("save_path", "./output")
    if is_main_process:
        os.makedirs(f"{save_path}/{run_name}", exist_ok=True)
        with open(f"{save_path}/{run_name}/config.yaml", "w") as f:
            yaml.dump(config, f)

    # Initialize WandB if configured
    wandb_config = training_config.get("wandb", None)
    if wandb_config is not None and is_main_process:
        if training_config.get("sample_interval", 1) > 10:
            init_wandb(wandb_config, run_name)

    # Log basic info
    print(f"Rank: {rank}")
    if is_main_process:
        print(f"Config: {config}")
        
    #
    # Dataset preparation
    #
    if training_config["dataset"]["type"] == "image_condition":
        # Load dataset from disk
        save_dir = training_config["dataset"].get(
            "path", "IPA300K-modified/arrow_dataset"
        )
        dataset = load_from_disk(save_dir)

        # Define filter function to exclude oversized bounding boxes
        def filter_fn(item):
            bbox = item["bbox"]
            if isinstance(bbox, str):
                bbox = ast.literal_eval(bbox)
            width = int(bbox[2]) - int(bbox[0])
            height = int(bbox[3]) - int(bbox[1])
            return not (width > 350 and height > 350)
        
        # Apply filter and cache results
        os.makedirs("./cache/dataset", exist_ok=True)
        dataset = dataset.filter(
            filter_fn, 
            cache_file_name="./cache/dataset/filtered_layer.arrow"
        )
        
        # Create dataset with appropriate transformations
        dataset = LayerSubjectDataset(
            dataset,
            condition_size=training_config["dataset"]["condition_size"],
            condition_size2=training_config["dataset"]["condition_size2"],
            target_size=training_config["dataset"]["target_size"],
            image_size=training_config["dataset"]["image_size"],
            condition_type1=training_config["condition_type1"],
            condition_type2=training_config["condition_type2"],
            drop_text_prob=training_config["dataset"]["drop_text_prob"],
            drop_image_prob=training_config["dataset"]["drop_image_prob"],
        )
    else:
        raise NotImplementedError(f"Dataset type '{training_config['dataset']['type']}' not supported")

    # Create DataLoader
    print(f"Dataset length: {len(dataset)}")
    train_loader = DataLoader(
        dataset,
        batch_size=training_config["batch_size"],
        shuffle=True,
        num_workers=training_config["dataloader_workers"],
        pin_memory=True,
    )
    
    #
    # Model initialization
    #
    
    # Check if we're resuming from checkpoint
    resume_from_checkpoint = training_config.get("resume_from_checkpoint", False)
    if resume_from_checkpoint:
        lora_path1 = training_config.get("lora_path1", None)
        lora_path2 = training_config.get("lora_path2", None)
        if lora_path1 is None or lora_path2 is None:
            raise ValueError("Please provide paths to both LoRA weights when resuming training")

    # Initialize model
    trainable_model = OINModel(
        flux_pipe_id=config["flux_path"],
        lora_config=training_config.get("lora_config", None),
        lora_path1=training_config.get("lora_path1") if resume_from_checkpoint else None,
        lora_path2=training_config.get("lora_path2") if resume_from_checkpoint else None,
        device="cuda",
        dtype=getattr(torch, config["dtype"]),
        optimizer_config=training_config["optimizer"],
        model_config=config.get("model", {}),
        gradient_checkpointing=training_config.get("gradient_checkpointing", False),
        data_config=training_config["dataset"],
    )
    
    #
    # Training setup
    #
    
    # Initialize callbacks for logging and checkpoints
    training_callbacks = (
        [TrainingCallback(run_name, training_config=training_config)]
        if is_main_process
        else []
    )

    # Configure trainer
    trainer = L.Trainer(
        accumulate_grad_batches=training_config["accumulate_grad_batches"],
        callbacks=training_callbacks,
        enable_checkpointing=False,  # We handle checkpointing in our callback
        enable_progress_bar=True,
        logger=False,  # We handle logging in our callback
        max_steps=training_config.get("max_steps", -1),
        max_epochs=training_config.get("max_epochs", -1),
        gradient_clip_val=training_config.get("gradient_clip_val", 0.5),
    )

    # Attach training config to trainer for callback access
    setattr(trainer, "training_config", training_config)

    # Start training
    trainer.fit(trainable_model, train_loader)


if __name__ == "__main__":
    main()