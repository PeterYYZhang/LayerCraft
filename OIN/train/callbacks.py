"""
Dual-Condition Training Callback Module

This module provides callbacks for monitoring, logging, checkpoint saving,
and sample generation during the training of dual-condition diffusion models.
It supports Weights & Biases integration, regular checkpoint saving, and
periodic sample generation to visualize training progress.
"""

import os
import json
import torch
import numpy as np
import lightning as L
from typing import Dict, List, Optional, Tuple, Union, Any
from PIL import Image, ImageFilter, ImageDraw

# Optional dependency - WandB
try:
    import wandb
except ImportError:
    wandb = None

from ..flux.condition import Condition
from ..flux.generate import generate


def enlarge_bbox(bbox: List[int]) -> List[int]:
    """
    Enlarge a bounding box with proportional padding.
    
    Args:
        bbox: Original bounding box coordinates [x1, y1, x2, y2]
        
    Returns:
        List of enlarged bounding box coordinates
    """
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    
    # Add padding: 5% on left and right, 15% on bottom
    bbox[0] = max(0, int(bbox[0] - width * 0.05))
    bbox[2] = min(512, int(bbox[2] + width * 0.05))
    bbox[3] = min(512, int(bbox[3] + height * 0.15))
    
    return bbox


def load_bbox(json_path: str) -> List[int]:
    """
    Load bounding box coordinates from a JSON file.
    
    Args:
        json_path: Path to the JSON file containing bounding box data
        
    Returns:
        List of bounding box coordinates [x1, y1, x2, y2]
    """
    with open(json_path, "r") as f:
        return json.load(f)[0]['bbox_xyxy']


def load_text(file_path: str) -> str:
    """
    Load text content from a file.
    
    Args:
        file_path: Path to the text file
        
    Returns:
        Text content of the file with whitespace stripped
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().strip()


class TrainingCallback(L.Callback):
    """
    PyTorch Lightning callback for monitoring training, saving checkpoints,
    and generating sample images during dual-condition diffusion model training.
    
    Attributes:
        run_name: Name of the current training run
        training_config: Configuration dictionary for training parameters
        print_every_n_steps: How often to print training metrics
        save_interval: How often to save model checkpoints
        sample_interval: How often to generate sample images
        save_path: Directory to save checkpoints and samples
        use_wandb: Whether to use Weights & Biases for logging
        total_steps: Counter for total training steps
    """
    
    def __init__(self, run_name: str, training_config: Dict[str, Any] = None):
        """
        Initialize the training callback.
        
        Args:
            run_name: Name of the current training run
            training_config: Configuration dictionary for training parameters
        """
        super().__init__()
        training_config = training_config or {}
        self.run_name = run_name
        self.training_config = training_config

        # Configure reporting and saving intervals
        self.print_every_n_steps = training_config.get("print_every_n_steps", 10)
        self.save_interval = training_config.get("save_interval", 1000)
        self.sample_interval = training_config.get("sample_interval", 1000)
        self.save_path = training_config.get("save_path", "./output")

        # Configure Weights & Biases integration
        self.wandb_config = training_config.get("wandb", None)
        self.use_wandb = (
            wandb is not None and os.environ.get("WANDB_API_KEY") is not None
        )

        # Initialize step counter
        self.total_steps = 0

    def on_train_batch_end(
        self, 
        trainer: L.Trainer, 
        pl_module: L.LightningModule, 
        outputs: Dict[str, torch.Tensor], 
        batch: Dict[str, Any], 
        batch_idx: int
    ) -> None:
        """
        Callback triggered at the end of each training batch.
        
        This method:
        1. Calculates and logs gradient metrics
        2. Logs training progress to Weights & Biases if configured
        3. Saves model checkpoints at specified intervals
        4. Generates sample images at specified intervals
        
        Args:
            trainer: PyTorch Lightning trainer
            pl_module: The model being trained
            outputs: Output from the training step
            batch: Current batch of data
            batch_idx: Index of the current batch
        """
        # Calculate gradient statistics
        gradient_size = 0
        max_gradient_size = 0
        count = 0
        for _, param in pl_module.named_parameters():
            if param.grad is not None:
                gradient_size += param.grad.norm(2).item()
                max_gradient_size = max(max_gradient_size, param.grad.norm(2).item())
                count += 1
        
        if count > 0:
            gradient_size /= count

        # Increment step counter
        self.total_steps += 1

        # Log metrics to Weights & Biases if configured
        if self.use_wandb and self.sample_interval > 10:
            report_dict = {
                "steps": self.total_steps,
                "epoch": trainer.current_epoch,
                "gradient_size": gradient_size,
                "max_gradient_size": max_gradient_size,
            }
            
            # Add loss metrics
            if "loss" in outputs:
                loss_value = outputs["loss"].item() * trainer.accumulate_grad_batches
                report_dict["loss"] = loss_value
            
            # Add timestep information if available
            if hasattr(pl_module, "last_t"):
                report_dict["t"] = pl_module.last_t
                
            # Log to WandB
            wandb.log(report_dict)

        # Print training progress at specified intervals
        if self.total_steps % self.print_every_n_steps == 0:
            print(
                f"Epoch: {trainer.current_epoch}, "
                f"Steps: {self.total_steps}, "
                f"Batch: {batch_idx}, "
                f"Loss: {pl_module.log_loss:.4f}, "
                f"Gradient size: {gradient_size:.4f}, "
                f"Max gradient size: {max_gradient_size:.4f}"
            )

        # Save model checkpoints at specified intervals
        if self.total_steps % self.save_interval == 0:
            print(
                f"Epoch: {trainer.current_epoch}, "
                f"Steps: {self.total_steps} - Saving LoRA weights"
            )
            
            # Create checkpoint directory if it doesn't exist
            checkpoint_dir = f"{self.save_path}/{self.run_name}/ckpt"
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # Save the first LoRA adapter (fill)
            pl_module.save_lora(
                f"{checkpoint_dir}/{self.total_steps}_fill",
                adapter_name="fill"
            )
            
            # Save the second LoRA adapter (subject)
            pl_module.save_lora(
                f"{checkpoint_dir}/{self.total_steps}_subject",
                adapter_name="subject"
            )

        # Generate sample images at specified intervals
        if self.total_steps % self.sample_interval == 0:
            print(
                f"Epoch: {trainer.current_epoch}, "
                f"Steps: {self.total_steps} - Generating samples"
            )
            
            # Create output directory if it doesn't exist
            output_dir = f"{self.save_path}/{self.run_name}/output"
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate samples using the current condition types from the batch
            self.generate_a_sample(
                trainer,
                pl_module,
                output_dir,
                f"lora_{self.total_steps}",
                batch["condition_type1"][0],
                batch["condition_type2"][0],
            )

    @torch.no_grad()
    def generate_a_sample(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        save_path: str,
        file_name: str,
        condition_type1: str = "subject_bg",
        condition_type2: str = "subject_obj",
    ) -> None:
        """
        Generate sample images during training to visualize progress.
        
        This method:
        1. Prepares test cases based on condition types
        2. Runs the model inference on each test case
        3. Saves the generated images, condition images, and prompts
        
        Args:
            trainer: PyTorch Lightning trainer
            pl_module: The model being trained
            save_path: Directory to save generated samples
            file_name: Base filename for the generated samples
            condition_type1: Type of first condition (default: "subject_bg")
            condition_type2: Type of second condition (default: "subject_obj")
        """
        # Get configuration parameters from training config
        condition_size = trainer.training_config["dataset"]["condition_size"]
        condition_size2 = trainer.training_config["dataset"]["condition_size2"]
        target_size = trainer.training_config["dataset"]["target_size"]

        # Set up deterministic generation for comparison across training
        generator = torch.Generator(device=pl_module.device)
        generator.manual_seed(42)

        # List to store test cases
        test_list = []

        # Prepare test cases based on condition types
        if condition_type1 == "fill" and condition_type2 == "subject":
            # Set up sample generation from the S200K dataset
            bg_path = "/scratch3/ck1_23/OminiControl/S200K-modified/filled_images"
            indices = [pth.split('.')[0] for pth in os.listdir(bg_path)]
            
            # Limit the number of samples to generate
            num_samples = min(1, len(indices))  # Only generating 1 sample for efficiency
            np.random.shuffle(indices)
            
            # Prepare test cases
            for i in range(num_samples):
                ind = str(indices[i])
                
                # Load assets for this sample
                description = load_text(f"S200K-modified/description/{ind}.txt")
                bbox = load_bbox(f"S200K-modified/bboxes/{ind}.json")
                bbox = enlarge_bbox(bbox.copy())
                bg = Image.open(f"S200K-modified/filled_images/{ind}.png")
                original_bg = bg.copy()
                
                # Create masked background
                bg.paste((0, 0, 0), (bbox[0], bbox[1], bbox[2], bbox[3]))
                
                # Load instance name and subject image
                instance = load_text(f"S200K-modified/instance/{ind}.txt")
                subject_obj = Image.open(f"S200K-modified/right_img/{ind}.png")
                
                # Create test prompt
                prompt = f"Fill the blank area in the given background with {instance}, with description: {description}"
                
                # Add to test list
                test_list.append((
                    bg,                   # Background image with masked area
                    subject_obj,          # Subject image
                    [0, 0],               # Position delta for background
                    [0, -32],             # Position delta for subject
                    bbox,                 # Bounding box coordinates
                    prompt,               # Generation prompt
                    original_bg,          # Original background without masking
                ))
        else:
            raise NotImplementedError(f"Sample generation not implemented for condition types {condition_type1} and {condition_type2}")

        # Create save directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)
        
        # Generate samples for each test case
        for i, (condition1, condition2, position_delta, position_delta2, bbox, prompt, original_bg) in enumerate(test_list):
            print(f"Generating sample {i+1} of {len(test_list)}")
            
            # Save copies of original condition images before resizing
            condition1_copy = condition1.copy()
            condition2_copy = condition2.copy()
            
            # Prepare condition objects
            condition1_obj = Condition(
                condition_type=condition_type1,
                condition=condition1.resize((condition_size, condition_size)).convert("RGB"),
                position_delta=position_delta,
            )
            
            condition2_obj = Condition(
                condition_type=condition_type2,
                condition=condition2.resize((condition_size2, condition_size2)).convert("RGB"),
                position_delta=position_delta2,
            )
            
            # Generate image
            res = generate(
                pl_module.flux_pipe,
                prompt=prompt,
                conditions=[condition1_obj],
                conditions2=[condition2_obj],
                height=target_size,
                width=target_size,
                generator=generator,
                model_config=pl_module.model_config,
                default_lora=True,
                num_inference_steps=50,
                bbox=bbox,
                latents_temp=condition1_copy.convert("RGB"),
            )
            
            # Save generated image and condition images
            res.images[0].save(os.path.join(save_path, f"{file_name}_layer_{i}.jpg"))
            condition1_copy.save(os.path.join(save_path, f"{file_name}_bg_{i}.png"))
            condition2_copy.save(os.path.join(save_path, f"{file_name}_cond_{i}.jpg"))
            original_bg.save(os.path.join(save_path, f"{file_name}_original_bg_{i}.png"))
            
            # Save the prompt
            with open(os.path.join(save_path, f"{file_name}_prompt_{i}.txt"), "w") as f:
                f.write(prompt)