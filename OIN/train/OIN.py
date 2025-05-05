"""
OminiModel: Dual-Condition LoRA Training Module

This module implements a PyTorch Lightning model for training dual-condition LoRA adapters
for the Flux diffusion pipeline. It enables fine-tuning the model to handle two different
conditioning signals simultaneously (e.g., background and subject), with spatial control
via masks.

The implementation supports:
- Loading and training multiple LoRA adapters simultaneously
- Training with mixed precision
- Spatial conditioning via bounding box masks
- Configurable optimization strategies
"""

import torch
import lightning as L
from typing import Dict, List, Optional, Union, Any, Tuple
from peft import LoraConfig, get_peft_model_state_dict
from diffusers.pipelines import FluxPipeline
import prodigyopt

from ..flux.OIN_transformer import tranformer_forward
from ..flux.condition import Condition
from ..flux.pipeline_tools import encode_images, prepare_text_input, downsample_bbox_to_mask, pack_mask


class OINModel(L.LightningModule):
    """
    PyTorch Lightning module for training dual-condition LoRA adapters for Flux.
    
    This model allows training two separate LoRA adapters that can be mixed during
    inference based on spatial masks, enabling region-specific conditioning.
    
    Attributes:
        flux_pipe: Loaded Flux pipeline
        transformer: Transformer model from the pipeline
        lora_layers: List of trainable LoRA parameters
        model_config: Configuration for model behavior
        optimizer_config: Configuration for optimization
        data_config: Configuration for data processing
    """
    
    def __init__(
        self,
        flux_pipe_id: str,
        lora_path1: Optional[str] = None,
        lora_path2: Optional[str] = None,
        lora_config: Optional[dict] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        model_config: Optional[Dict[str, Any]] = None,
        optimizer_config: Optional[Dict[str, Any]] = None,
        gradient_checkpointing: bool = False,
        data_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the OminiModel for dual-condition training.
        
        Args:
            flux_pipe_id: ID or path of the pretrained Flux model
            lora_path1: Path to the first LoRA adapter weights (fill/background)
            lora_path2: Path to the second LoRA adapter weights (subject)
            lora_config: Configuration for creating new LoRA adapters
            device: Device to run the model on ("cuda", "cpu", etc.)
            dtype: Data type for model parameters (e.g., torch.bfloat16)
            model_config: Configuration parameters for model behavior
            optimizer_config: Configuration for the optimizer
            gradient_checkpointing: Whether to use gradient checkpointing
            data_config: Configuration for data processing
        """
        # Initialize the LightningModule
        super().__init__()
        self.model_config = model_config or {}
        self.optimizer_config = optimizer_config or {}
        self.data_config = data_config or {"type": "image_condition"}

        # Load the Flux pipeline
        self.flux_pipe: FluxPipeline = (
            FluxPipeline.from_pretrained(flux_pipe_id).to(dtype=dtype).to(device)
        )
        self.transformer = self.flux_pipe.transformer
        self.transformer.gradient_checkpointing = gradient_checkpointing
        self.transformer.train()

        # Freeze the encoders and VAE (we only train the LoRA layers)
        self.flux_pipe.text_encoder.requires_grad_(False).eval()
        self.flux_pipe.text_encoder_2.requires_grad_(False).eval()
        self.flux_pipe.vae.requires_grad_(False).eval()

        # Initialize LoRA layers
        self.lora_layers = self.init_lora(lora_path1, lora_path2, lora_config)

        # Move model to specified device and dtype
        self.to(device).to(dtype)
        
        # Logging tracking
        self.log_loss = None
        self.last_t = None

    def init_lora(
        self, 
        lora_path1: Optional[str] = None, 
        lora_path2: Optional[str] = None, 
        lora_config: Optional[Dict[str, Any]] = None
    ) -> List[torch.nn.Parameter]:
        """
        Initialize LoRA adapters for the model.
        
        This method either:
        1. Loads pre-trained LoRA weights from specified paths, or
        2. Creates new LoRA adapters based on the provided configuration
        
        Args:
            lora_path1: Path to the first LoRA adapter weights (fill/background)
            lora_path2: Path to the second LoRA adapter weights (subject)
            lora_config: Configuration for creating new LoRA adapters
            
        Returns:
            List of trainable LoRA parameters
            
        Raises:
            AssertionError: If no LoRA source (path or config) is provided
        """
        # Ensure at least one option is provided
        assert lora_path1 or lora_path2 or lora_config, "At least one of lora_path1, lora_path2, or lora_config must be provided"
        
        # Case 1: Using pre-trained LoRA weights
        if lora_path1 or lora_path2:
            # Load first adapter if path is provided (fill/background adapter)
            if lora_path1:
                self.flux_pipe.load_lora_weights(
                    lora_path1,
                    adapter_name="fill",
                )
                self.log(f"Loaded first LoRA adapter (fill) from {lora_path1}")
            
            # Load second adapter if path is provided (subject adapter)
            if lora_path2:
                self.flux_pipe.load_lora_weights(
                    lora_path2,
                    adapter_name="subject",
                )
                self.log(f"Loaded second LoRA adapter (subject) from {lora_path2}")
            
            # Set default adapter if only one is loaded
            if lora_path1 and not lora_path2:
                # Only first adapter loaded, set it as active
                self.transformer.add_adapter("fill")
            elif lora_path2 and not lora_path1:
                # Only second adapter loaded, set it as active
                self.transformer.add_adapter("subject")
            
        # Case 2: Creating new adapters from config
        elif lora_config:
            self.transformer.add_adapter("fill", LoraConfig(**lora_config))
            self.transformer.add_adapter("subject", LoraConfig(**lora_config))
            self.log("Created new LoRA adapters from config")
        
        # Get trainable parameters
        lora_layers = list(filter(lambda p: p.requires_grad, self.transformer.parameters()))
        self.log(f"Found {len(lora_layers)} trainable LoRA parameters")
        
        return lora_layers

    def save_lora(self, path: str, adapter_name: Optional[str] = None) -> None:
        """
        Save LoRA weights to the specified path.
        
        Args:
            path: Base path where LoRA weights should be saved
            adapter_name: Name of the specific adapter to save.
                If None, saves the model with all adapters.
        """
        if adapter_name is not None:
            # Get state dict for a specific adapter
            state_dict = get_peft_model_state_dict(self.transformer, adapter_name=adapter_name)
            # Modify path to include adapter name to avoid overwriting
            adapter_path = f"{path}_{adapter_name}"
            FluxPipeline.save_lora_weights(
                save_directory=adapter_path,
                transformer_lora_layers=state_dict,
                safe_serialization=True,
            )
            self.log(f"Saved LoRA adapter '{adapter_name}' to {adapter_path}")
        else:
            # Save all adapters
            FluxPipeline.save_lora_weights(
                save_directory=path,
                transformer_lora_layers=get_peft_model_state_dict(self.transformer),
                safe_serialization=True,
            )
            self.log(f"Saved all LoRA adapters to {path}")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configure optimizers for training.
        
        This method sets up the optimizer based on the optimizer_config,
        ensuring only LoRA parameters are trained while the base model remains frozen.
        
        Returns:
            Configured optimizer instance
        
        Raises:
            NotImplementedError: If an unsupported optimizer type is specified
        """
        # Freeze the transformer
        self.transformer.requires_grad_(False)
        opt_config = self.optimizer_config

        # Set the trainable parameters
        self.trainable_params = self.lora_layers

        # Unfreeze trainable parameters
        for p in self.trainable_params:
            p.requires_grad_(True)

        # Initialize the optimizer based on config
        if opt_config["type"] == "AdamW":
            optimizer = torch.optim.AdamW(self.trainable_params, **opt_config["params"])
        elif opt_config["type"] == "Prodigy":
            optimizer = prodigyopt.Prodigy(
                self.trainable_params,
                **opt_config["params"],
            )
        elif opt_config["type"] == "SGD":
            optimizer = torch.optim.SGD(self.trainable_params, **opt_config["params"])
        else:
            raise NotImplementedError(f"Optimizer type {opt_config['type']} not supported")

        return optimizer

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """
        Execute a single training step.
        
        This method is called by PyTorch Lightning during training.
        
        Args:
            batch: Dictionary containing the current batch of data
            batch_idx: Index of the current batch
            
        Returns:
            Loss tensor for backpropagation
        """
        step_loss = self.step(batch)
        
        # Exponential moving average for loss logging
        self.log_loss = (
            step_loss.item()
            if not hasattr(self, "log_loss") or self.log_loss is None
            else self.log_loss * 0.95 + step_loss.item() * 0.05
        )
        
        # Log metrics
        self.log("train_loss", step_loss, prog_bar=True)
        if hasattr(self, "last_t") and self.last_t is not None:
            self.log("timestep", self.last_t, prog_bar=True)
            
        return step_loss

    def step(self, batch: Dict[str, Any]) -> torch.Tensor:
        """
        Process a batch and calculate the loss.
        
        This method handles the forward pass of the model and loss calculation.
        It supports different data configurations based on self.data_config.
        
        Args:
            batch: Dictionary containing the batch data
            
        Returns:
            Loss tensor
            
        Raises:
            NotImplementedError: If the data configuration is not supported
        """
        # Extract data based on data configuration
        if self.data_config["type"] == "image_condition":
            imgs = batch["image"]                    # Target images
            bg = batch["bg"]                        # Condition 1: background with mask
            conditions = batch["condition"]         # Condition 2: subject
            condition_types = batch["condition_type1"]   # Type of first condition
            condition_types2 = batch["condition_type2"]  # Type of second condition
            prompts = batch["prompt"]               # Text prompts
            bboxes = batch["bbox"]                  # Bounding boxes for masking
            position_delta = batch["position_delta1"][0]  # Position embed for condition 1
            position_delta2 = batch["position_delta2"][0] # Position embed for condition 2
        else:
            raise NotImplementedError(f"Data configuration type {self.data_config['type']} is not supported")
        
        # Prepare inputs (no gradient needed for these operations)
        with torch.no_grad():
            # Encode target images
            x_0, img_ids = encode_images(self.flux_pipe, imgs)
            
            # Create spatial mask from bounding boxes
            imgs_latent = encode_images(self.flux_pipe, imgs, to_vae=True)
            mask = downsample_bbox_to_mask(imgs_latent, bboxes)
            pack_masks = pack_mask(mask)

            # Encode text prompts
            prompt_embeds, pooled_prompt_embeds, text_ids = prepare_text_input(
                self.flux_pipe, prompts
            )
            
            # Generate noisy latents at random timestep t
            t = torch.sigmoid(torch.randn((imgs.shape[0],), device=self.device))
            x_1 = torch.randn_like(x_0).to(self.device)
            t_ = t.unsqueeze(1).unsqueeze(1)
            x_t = ((1 - t_) * x_0 + t_ * x_1).to(self.dtype)
            
            # Encode condition images
            condition_latents, condition_ids = encode_images(self.flux_pipe, bg)
            condition_latents2, condition_ids2 = encode_images(self.flux_pipe, conditions)
            
            # Apply position adjustments to conditions
            condition_ids[:, 1] += position_delta[0]
            condition_ids[:, 2] += position_delta[1]
            condition_ids2[:, 1] += position_delta2[0]
            condition_ids2[:, 2] += position_delta2[1]

            # Create condition type ID tensors
            condition_type_ids = torch.tensor(
                [Condition.get_type_id(ctype) for ctype in condition_types]
            ).to(self.device)
            condition_type_ids = (
                torch.ones_like(condition_ids[:, 0]) * condition_type_ids[0]
            ).unsqueeze(1)

            # Prepare guidance signal if configured
            guidance = (
                torch.ones_like(t).to(self.device)
                if self.transformer.config.guidance_embeds
                else None
            )
            
        # Forward pass through transformer
        transformer_out = tranformer_forward(
            self.transformer,
            # Model config
            model_config=self.model_config,
            # Spatial mask for attention
            latent_mask=pack_masks,
            # First condition inputs
            condition_latents=condition_latents,
            condition_ids=condition_ids,
            condition_type_ids=condition_type_ids,
            # Second condition inputs
            condition_latents2=condition_latents2,
            condition_ids2=condition_ids2,
            # Standard transformer inputs
            hidden_states=x_t,
            timestep=t,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=img_ids,
            joint_attention_kwargs=None,
            return_dict=False,
        )
        pred = transformer_out[0]

        # Compute loss on the noise prediction
        loss = torch.nn.functional.mse_loss(pred, (x_1 - x_0))
        loss = loss.mean()
        
        # Store current timestep for logging
        self.last_t = t.mean().item()
        
        return loss