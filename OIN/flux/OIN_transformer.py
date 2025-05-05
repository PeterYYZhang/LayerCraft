"""
Mixed Condition Transformer Implementation

This module extends the Flux transformer model to support dual conditional inputs.
It provides functionality for processing multiple conditioning signals (e.g., text and images)
within the same transformer architecture, enabling controlled generation based on
multiple reference inputs.
"""

import torch
import numpy as np
from typing import List, Union, Optional, Dict, Any, Callable, Tuple

from diffusers.pipelines import FluxPipeline
from diffusers.models.transformers.transformer_flux import (
    FluxTransformer2DModel,
    Transformer2DModelOutput,
    USE_PEFT_BACKEND,
    is_torch_version,
    scale_lora_layers,
    unscale_lora_layers,
    logger,
)

from .OIN_block import block_forward, single_block_forward
from .lora_controller import enable_lora
from .pipeline_tools import permute_by_mask, repermute_by_mask


def prepare_params(
    hidden_states: torch.Tensor,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    pooled_projections: Optional[torch.Tensor] = None,
    timestep: Optional[torch.LongTensor] = None,
    img_ids: Optional[torch.Tensor] = None,
    txt_ids: Optional[torch.Tensor] = None,
    guidance: Optional[torch.Tensor] = None,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    controlnet_block_samples: Optional[List[torch.Tensor]] = None,
    controlnet_single_block_samples: Optional[List[torch.Tensor]] = None,
    return_dict: bool = True,
    **kwargs: dict,
) -> Tuple:
    """
    Prepare and standardize parameters for transformer forward pass.
    
    This function collects and returns the essential parameters needed for
    the transformer's forward pass, providing a clean interface between
    different components.
    
    Args:
        hidden_states: Input latent states to be processed
        encoder_hidden_states: Text or context encoder states
        pooled_projections: Pooled embeddings for conditioning
        timestep: Current diffusion timestep
        img_ids: Position IDs for image tokens
        txt_ids: Position IDs for text tokens
        guidance: Optional guidance signal for controlled generation
        joint_attention_kwargs: Additional attention parameters
        controlnet_block_samples: Optional controlnet residuals for transformer blocks
        controlnet_single_block_samples: Optional controlnet residuals for single transformer blocks
        return_dict: Whether to return output as a dictionary
        kwargs: Additional keyword arguments
        
    Returns:
        Tuple of standardized parameters
    """
    return (
        hidden_states,
        encoder_hidden_states,
        pooled_projections,
        timestep,
        img_ids,
        txt_ids,
        guidance,
        joint_attention_kwargs,
        controlnet_block_samples,
        controlnet_single_block_samples,
        return_dict,
    )


def tranformer_forward(
    transformer: FluxTransformer2DModel,
    condition_latents: Optional[torch.Tensor],
    condition_ids: Optional[torch.Tensor],
    condition_type_ids: Optional[torch.Tensor],
    condition_latents2: Optional[torch.Tensor] = None,
    condition_ids2: Optional[torch.Tensor] = None,
    condition_type_ids2: Optional[torch.Tensor] = None,
    latent_mask: Optional[torch.Tensor] = None,
    model_config: Optional[Dict[str, Any]] = None,
    c_t: float = 0,
    **params: dict,
) -> Transformer2DModelOutput:
    """
    Extended forward pass for FluxTransformer2DModel supporting dual conditional inputs.
    
    This implementation allows the transformer to process two different conditional inputs
    (e.g., two different images or concepts) and mix them according to a spatial mask.
    
    Args:
        transformer: FluxTransformer2DModel instance
        condition_latents: First conditional input latents
        condition_ids: Position IDs for first condition
        condition_type_ids: Type IDs for first condition
        condition_latents2: Second conditional input latents
        condition_ids2: Position IDs for second condition
        condition_type_ids2: Type IDs for second condition
        latent_mask: Spatial mask determining where each condition applies
        model_config: Configuration options for model behavior
        c_t: Conditioning timestep (between 0 and 1)
        **params: Additional parameters for the transformer
        
    Returns:
        Transformer2DModelOutput containing the processed output
    """
    self = transformer
    if model_config is None:
        model_config = {}
        
    # Check if both conditions are provided
    use_condition = condition_latents is not None and condition_latents2 is not None
    
    # Unpack parameters
    (
        hidden_states,
        encoder_hidden_states,
        pooled_projections,
        timestep,
        img_ids,
        txt_ids,
        guidance,
        joint_attention_kwargs,
        controlnet_block_samples,
        controlnet_single_block_samples,
        return_dict,
    ) = prepare_params(**params)

    # Handle LoRA scaling
    if joint_attention_kwargs is not None:
        joint_attention_kwargs = joint_attention_kwargs.copy()
        lora_scale = joint_attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0
    
    if USE_PEFT_BACKEND:
        # Weight the LoRA layers by setting `lora_scale` for each PEFT layer
        scale_lora_layers(self, lora_scale)
    else:
        if (
            joint_attention_kwargs is not None
            and joint_attention_kwargs.get("scale", None) is not None
        ):
            logger.warning(
                "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
            )

    # Embed input latents
    with enable_lora((self.x_embedder,), model_config.get("latent_lora", False)):
        hidden_states = self.x_embedder(hidden_states)
    
    # Embed first conditional latents if provided
    if use_condition:
        with enable_lora((self.context_embedder,), True, 'fill'):
            condition_latents = self.x_embedder(condition_latents)
    
        # Embed second conditional latents
        with enable_lora((self.context_embedder,), True, 'subject'):
            condition_latents2 = self.x_embedder(condition_latents2)

    # Process timestep and guidance
    timestep = timestep.to(hidden_states.dtype) * 1000
    
    if guidance is not None:
        guidance = guidance.to(hidden_states.dtype) * 1000
    
    # Generate timestep embeddings
    temb = (
        self.time_text_embed(timestep, pooled_projections)
        if guidance is None
        else self.time_text_embed(timestep, guidance, pooled_projections)
    )

    # Generate conditional timestep embeddings
    cond_temb = None
    if use_condition:
        cond_temb = (
            self.time_text_embed(torch.ones_like(timestep) * c_t * 1000, pooled_projections)
            if guidance is None
            else self.time_text_embed(
                torch.ones_like(timestep) * c_t * 1000, guidance, pooled_projections
            )
        )
    
    # Embed encoder hidden states
    encoder_hidden_states = self.context_embedder(encoder_hidden_states)

    # Handle position IDs
    if txt_ids.ndim == 3:
        logger.warning(
            "Passing `txt_ids` 3d torch.Tensor is deprecated. "
            "Please remove the batch dimension and pass it as a 2d torch Tensor"
        )
        txt_ids = txt_ids[0]
        
    if img_ids.ndim == 3:
        logger.warning(
            "Passing `img_ids` 3d torch.Tensor is deprecated. "
            "Please remove the batch dimension and pass it as a 2d torch Tensor"
        )
        img_ids = img_ids[0]
        
    if img_ids.ndim == 3:
        img_ids = img_ids.squeeze(0)
        
    # Concatenate text and image position IDs
    ids = torch.cat((txt_ids, img_ids), dim=0)
    
    # Generate positional embeddings
    image_rotary_emb = self.pos_embed(ids)
    
    # Generate conditional positional embeddings if using conditions
    cond_rotary_emb = None
    cond_rotary_emb2 = None
    if use_condition:
        cond_rotary_emb = self.pos_embed(condition_ids)
        cond_rotary_emb2 = self.pos_embed(condition_ids2)

    #
    # Process through transformer blocks
    #
    for index_block, block in enumerate(self.transformer_blocks):
        if self.training and self.gradient_checkpointing:
            # Use gradient checkpointing for memory efficiency during training
            ckpt_kwargs: Dict[str, Any] = (
                {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            )
            encoder_hidden_states, hidden_states, condition_latents, condition_latents2 = (
                torch.utils.checkpoint.checkpoint(
                    block_forward,
                    self=block,
                    model_config=model_config,
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    condition_latents=condition_latents if use_condition else None,
                    condition_latents2=condition_latents2 if use_condition else None,
                    temb=temb,
                    cond_temb=cond_temb,
                    cond_rotary_emb=cond_rotary_emb,
                    cond_rotary_emb2=cond_rotary_emb2,
                    image_rotary_emb=image_rotary_emb,
                    latent_mask=latent_mask,
                    **ckpt_kwargs,
                )
            )
        else:
            # Standard forward pass
            encoder_hidden_states, hidden_states, condition_latents, condition_latents2 = block_forward(
                block,
                model_config=model_config,
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                condition_latents=condition_latents if use_condition else None,
                condition_latents2=condition_latents2 if use_condition else None,
                temb=temb,
                cond_temb=cond_temb,
                cond_rotary_emb=cond_rotary_emb,
                cond_rotary_emb2=cond_rotary_emb2,
                image_rotary_emb=image_rotary_emb,
                latent_mask=latent_mask,
            )

        # Apply controlnet residual if provided
        if controlnet_block_samples is not None:
            interval_control = len(self.transformer_blocks) / len(
                controlnet_block_samples
            )
            interval_control = int(np.ceil(interval_control))
            hidden_states = (
                hidden_states
                + controlnet_block_samples[index_block // interval_control]
            )
    
    # Concatenate encoder and hidden states for single transformer blocks
    hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

    #
    # Process through single transformer blocks
    #
    for index_block, block in enumerate(self.single_transformer_blocks):
        if self.training and self.gradient_checkpointing:
            # Use gradient checkpointing for memory efficiency during training
            ckpt_kwargs: Dict[str, Any] = (
                {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            )
            result = torch.utils.checkpoint.checkpoint(
                single_block_forward,
                self=block,
                model_config=model_config,
                hidden_states=hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                latent_mask=latent_mask,
                **(
                    {
                        "condition_latents": condition_latents,
                        "condition_latents2": condition_latents2,
                        "cond_temb": cond_temb,
                        "cond_rotary_emb": cond_rotary_emb,
                        "cond_rotary_emb2": cond_rotary_emb2,
                    }
                    if use_condition
                    else {}
                ),
                **ckpt_kwargs,
            )
        else:
            # Standard forward pass
            result = single_block_forward(
                block,
                model_config=model_config,
                hidden_states=hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                latent_mask=latent_mask,
                **(
                    {
                        "condition_latents": condition_latents,
                        "condition_latents2": condition_latents2,
                        "cond_temb": cond_temb,
                        "cond_rotary_emb": cond_rotary_emb,
                        "cond_rotary_emb2": cond_rotary_emb2,
                    }
                    if use_condition
                    else {}
                ),
            )
            
        # Unpack results based on whether conditions were used
        if use_condition:
            hidden_states, condition_latents, condition_latents2 = result
        else:
            hidden_states = result

        # Apply controlnet residual if provided
        if controlnet_single_block_samples is not None:
            interval_control = len(self.single_transformer_blocks) / len(
                controlnet_single_block_samples
            )
            interval_control = int(np.ceil(interval_control))
            hidden_states[:, encoder_hidden_states.shape[1]:, ...] = (
                hidden_states[:, encoder_hidden_states.shape[1]:, ...]
                + controlnet_single_block_samples[index_block // interval_control]
            )

    # Extract only the main hidden states (not encoder states)
    hidden_states = hidden_states[:, encoder_hidden_states.shape[1]:, ...]

    # Final normalization and projection
    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)

    # Clean up LoRA scaling if using PEFT backend
    if USE_PEFT_BACKEND:
        unscale_lora_layers(self, lora_scale)

    # Return appropriate output format
    if not return_dict:
        return (output,)
        
    return Transformer2DModelOutput(sample=output)