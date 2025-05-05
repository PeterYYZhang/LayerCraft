"""
Mixed Condition Attention Module

This module provides implementations of attention mechanisms and transformer blocks
that support multiple conditional inputs. It enables the model to process and mix
information from different conditioning sources (e.g., text, images, or latent representations)
during the generation process.
"""

import torch
from typing import List, Union, Optional, Dict, Any, Callable, Tuple
from diffusers.models.attention_processor import Attention, F
from .lora_controller import enable_lora
from .pipeline_tools import permute_by_mask, repermute_by_mask


def attn_forward(
    attn: Attention,
    hidden_states: torch.FloatTensor,
    encoder_hidden_states: Optional[torch.FloatTensor] = None,
    condition_latents: Optional[torch.FloatTensor] = None,
    condition_latents2: Optional[torch.FloatTensor] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    image_rotary_emb: Optional[torch.Tensor] = None,
    cond_rotary_emb: Optional[torch.Tensor] = None,
    cond_rotary_emb2: Optional[torch.Tensor] = None,
    model_config: Optional[Dict[str, Any]] = None,
    latent_mask: Optional[torch.Tensor] = None,
) -> Union[torch.FloatTensor, Tuple[torch.FloatTensor, ...]]:
    """
    Attention forward pass with support for multiple conditional inputs.
    
    This implementation supports two conditional latent inputs that can be mixed
    based on a latent mask. This enables region-specific conditioning in the
    generated output.
    
    Args:
        attn: Attention module
        hidden_states: Input hidden states
        encoder_hidden_states: Optional encoder hidden states (e.g., from text encoder)
        condition_latents: First conditional latent input
        condition_latents2: Second conditional latent input
        attention_mask: Optional attention mask
        image_rotary_emb: Optional rotary embeddings for image tokens
        cond_rotary_emb: Optional rotary embeddings for first condition
        cond_rotary_emb2: Optional rotary embeddings for second condition
        model_config: Optional model configuration dictionary
        latent_mask: Optional mask for mixing latent conditions (True values use condition2)
        
    Returns:
        Either a single tensor of processed hidden states,
        or a tuple containing combinations of (hidden_states, encoder_hidden_states,
        condition_latents, condition_latents2) depending on which inputs were provided
    """
    if model_config is None:
        model_config = {}
        
    batch_size, _, _ = (
        hidden_states.shape
        if encoder_hidden_states is None
        else encoder_hidden_states.shape
    )

    with enable_lora(
        (attn.to_q, attn.to_k, attn.to_v), model_config.get("latent_lora", False)
    ):
        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

    inner_dim = key.shape[-1]
    head_dim = inner_dim // attn.heads

    query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

    if attn.norm_q is not None:
        query = attn.norm_q(query)
    if attn.norm_k is not None:
        key = attn.norm_k(key)

    # Process encoder hidden states if provided
    if encoder_hidden_states is not None:
        # `context` projections.
        encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
        encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

        encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)
        encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)
        encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)

        if attn.norm_added_q is not None:
            encoder_hidden_states_query_proj = attn.norm_added_q(
                encoder_hidden_states_query_proj
            )
        if attn.norm_added_k is not None:
            encoder_hidden_states_key_proj = attn.norm_added_k(
                encoder_hidden_states_key_proj
            )

        # Concatenate encoder projections with hidden state projections
        query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
        key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
        value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

    # Apply rotary embeddings to image tokens if provided
    if image_rotary_emb is not None:
        from diffusers.models.embeddings import apply_rotary_emb
        query = apply_rotary_emb(query, image_rotary_emb)
        key = apply_rotary_emb(key, image_rotary_emb)

    # Process first conditional latent if provided
    if condition_latents is not None:
        with enable_lora((attn.to_q, attn.to_k, attn.to_v), True):
            cond_query = attn.to_q(condition_latents)
            cond_key = attn.to_k(condition_latents)
            cond_value = attn.to_v(condition_latents)

        cond_query = cond_query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        cond_key = cond_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        cond_value = cond_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        
        if attn.norm_q is not None:
            cond_query = attn.norm_q(cond_query)
        if attn.norm_k is not None:
            cond_key = attn.norm_k(cond_key)

    # Process second conditional latent if provided
    if condition_latents2 is not None:
        with enable_lora((attn.to_q, attn.to_k, attn.to_v), True):
            cond_query2 = attn.to_q(condition_latents2)
            cond_key2 = attn.to_k(condition_latents2)
            cond_value2 = attn.to_v(condition_latents2)

        cond_query2 = cond_query2.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        cond_key2 = cond_key2.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        cond_value2 = cond_value2.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        
        if attn.norm_q is not None:
            cond_query2 = attn.norm_q(cond_query2)
        if attn.norm_k is not None:
            cond_key2 = attn.norm_k(cond_key2)

    # Apply rotary embeddings to conditional latents if provided
    if cond_rotary_emb is not None and condition_latents is not None:
        from diffusers.models.embeddings import apply_rotary_emb
        cond_query = apply_rotary_emb(cond_query, cond_rotary_emb)
        cond_key = apply_rotary_emb(cond_key, cond_rotary_emb)

    if cond_rotary_emb2 is not None and condition_latents2 is not None:
        from diffusers.models.embeddings import apply_rotary_emb
        cond_query2 = apply_rotary_emb(cond_query2, cond_rotary_emb2)
        cond_key2 = apply_rotary_emb(cond_key2, cond_rotary_emb2)

    # Save base query/key/value for use with multiple conditions
    base_query = query.clone()
    base_key = key.clone()
    base_value = value.clone()

    # Combine base states with first conditional latent
    if condition_latents is not None:
        query = torch.cat([base_query, cond_query], dim=2)
        key = torch.cat([base_key, cond_key], dim=2)
        value = torch.cat([base_value, cond_value], dim=2)
    
    # Combine base states with second conditional latent
    if condition_latents2 is not None:
        query2 = torch.cat([base_query, cond_query2], dim=2)
        key2 = torch.cat([base_key, cond_key2], dim=2)
        value2 = torch.cat([base_value, cond_value2], dim=2)

    # Apply attention masking if needed
    if not model_config.get("union_cond_attn", True) and condition_latents is not None:
        # Mask attention between hidden states and condition latents
        attention_mask = torch.ones(
            query.shape[2], key.shape[2], device=query.device, dtype=torch.bool
        )
        condition_n = cond_query.shape[2]
        attention_mask[-condition_n:, :-condition_n] = False
        attention_mask[:-condition_n, -condition_n:] = False
        
    # Apply scaling bias if c_factor attribute exists
    if hasattr(attn, "c_factor") and condition_latents is not None:
        attention_mask = torch.zeros(
            query.shape[2], key.shape[2], device=query.device, dtype=query.dtype
        )
        condition_n = cond_query.shape[2]
        bias = torch.log(attn.c_factor[0])
        attention_mask[-condition_n:, :-condition_n] = bias
        attention_mask[:-condition_n, -condition_n:] = bias
    
    # Compute attention for first condition
    hidden_states = F.scaled_dot_product_attention(
        query, key, value, dropout_p=0.0, is_causal=False, attn_mask=attention_mask
    )
    
    # Compute attention for second condition
    if condition_latents2 is not None:
        hidden_states2 = F.scaled_dot_product_attention(
            query2, key2, value2, dropout_p=0.0, is_causal=False, attn_mask=attention_mask
        )

    # Reshape and convert to correct dtype
    hidden_states = hidden_states.transpose(1, 2).reshape(
        batch_size, -1, attn.heads * head_dim
    )
    hidden_states = hidden_states.to(query.dtype)

    if condition_latents2 is not None:
        hidden_states2 = hidden_states2.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states2 = hidden_states2.to(query2.dtype)

    # Handle the case with encoder hidden states
    if encoder_hidden_states is not None:
        if condition_latents is not None and condition_latents2 is not None:
            # Split hidden states into encoder, main, and condition components
            encoder_hidden_states, hidden_states, condition_latents = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[
                    :, encoder_hidden_states.shape[1] : -condition_latents.shape[1]
                ],
                hidden_states[:, -condition_latents.shape[1] :],
            )
            encoder_hidden_states2, hidden_states2, condition_latents2 = (
                hidden_states2[:, : encoder_hidden_states.shape[1]],
                hidden_states2[
                    :, encoder_hidden_states.shape[1] : -condition_latents2.shape[1]
                ],
                hidden_states2[:, -condition_latents2.shape[1] :],
            )
            
            # Mix encoder hidden states with fixed weight (0.5)
            encoder_hidden_states = encoder_hidden_states * 0.5 + encoder_hidden_states2 * 0.5
            
            # Mix main hidden states based on latent mask
            if latent_mask is not None:
                latent_mask = latent_mask.view(batch_size, -1, 1)
                hidden_states = (~latent_mask) * hidden_states + latent_mask * hidden_states2
            
        elif condition_latents is not None:
            # Split hidden states when only first condition is present
            encoder_hidden_states, hidden_states, condition_latents = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
                None,
            )
        elif condition_latents2 is not None:
            # Split hidden states when only second condition is present
            encoder_hidden_states, hidden_states, condition_latents2 = (
                hidden_states2[:, : encoder_hidden_states.shape[1]],
                hidden_states2[:, encoder_hidden_states.shape[1] :],
                None,
            )
        else:
            # Split hidden states when no conditions are present
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )

        # Apply output projections
        with enable_lora((attn.to_out[0],), model_config.get("latent_lora", False)):
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
            
        encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        # Apply output projections to first condition
        if condition_latents is not None:
            with enable_lora((attn.to_out[0],), True):
                condition_latents = attn.to_out[0](condition_latents)
                condition_latents = attn.to_out[1](condition_latents)
        
        # Apply output projections to second condition
        if condition_latents2 is not None:
            with enable_lora((attn.to_out[0],), True):
                condition_latents2 = attn.to_out[0](condition_latents2)
                condition_latents2 = attn.to_out[1](condition_latents2)
        
        # Return appropriate combination of outputs
        if condition_latents is not None and condition_latents2 is not None:
            return hidden_states, encoder_hidden_states, condition_latents, condition_latents2
        elif condition_latents is not None:
            return hidden_states, encoder_hidden_states, condition_latents, None
        elif condition_latents2 is not None:
            return hidden_states, encoder_hidden_states, None, condition_latents2
        else:
            return hidden_states, encoder_hidden_states
            
    # Handle the case without encoder hidden states
    elif condition_latents is not None:
        if condition_latents2 is not None:
            # Split hidden states for both conditions
            hidden_states, condition_latents = (
                hidden_states[:, :-condition_latents.shape[1]],
                hidden_states[:, -condition_latents.shape[1]:],
            )
            
            hidden_states2, condition_latents2 = (
                hidden_states2[:, :-condition_latents2.shape[1]],
                hidden_states2[:, -condition_latents2.shape[1]:],
            )
            
            # Mix hidden states with fixed weight (0.5)
            hidden_states = hidden_states * 0.5 + hidden_states2 * 0.5
            
            return hidden_states, condition_latents, condition_latents2
        else:
            # Split hidden states for only first condition
            hidden_states, condition_latents = (
                hidden_states[:, :-condition_latents.shape[1]],
                hidden_states[:, -condition_latents.shape[1]:],
            )
            
            # Apply output projections
            with enable_lora((attn.to_out[0],), model_config.get("latent_lora", False)):
                hidden_states = attn.to_out[0](hidden_states)
                hidden_states = attn.to_out[1](hidden_states)
            
            with enable_lora((attn.to_out[0],), True):
                condition_latents = attn.to_out[0](condition_latents)
                condition_latents = attn.to_out[1](condition_latents)
            
            return hidden_states, condition_latents, None
    else:
        return hidden_states


def block_forward(
    self,
    hidden_states: torch.FloatTensor,
    encoder_hidden_states: torch.FloatTensor,
    temb: torch.FloatTensor,
    condition_latents: Optional[torch.FloatTensor] = None,
    condition_latents2: Optional[torch.FloatTensor] = None,
    cond_temb: Optional[torch.FloatTensor] = None,
    cond_rotary_emb: Optional[torch.Tensor] = None,
    cond_rotary_emb2: Optional[torch.Tensor] = None,
    image_rotary_emb: Optional[torch.Tensor] = None,
    model_config: Optional[Dict[str, Any]] = None,
    latent_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.FloatTensor, ...]:
    """
    Forward pass for a transformer block with support for multiple conditional inputs.
    
    This implementation processes hidden states, encoder states, and up to two conditional
    latent inputs in parallel, with appropriate normalization and residual connections.
    
    Args:
        self: Block module instance
        hidden_states: Input hidden states
        encoder_hidden_states: Encoder hidden states (e.g., from text encoder)
        temb: Timestep embeddings
        condition_latents: First conditional latent input
        condition_latents2: Second conditional latent input
        cond_temb: Timestep embeddings for conditional latents
        cond_rotary_emb: Rotary embeddings for first condition
        cond_rotary_emb2: Rotary embeddings for second condition
        image_rotary_emb: Rotary embeddings for image
        model_config: Model configuration dictionary
        latent_mask: Mask for mixing latent conditions
        
    Returns:
        Tuple of (encoder_hidden_states, hidden_states, condition_latents, condition_latents2)
        where some elements may be None if corresponding inputs were not provided
    """
    if model_config is None:
        model_config = {}
        
    use_cond = condition_latents is not None and condition_latents2 is not None
    
    # Normalize hidden states with modulation based on timestep embeddings
    with enable_lora((self.norm1.linear,), model_config.get("latent_lora", False)):
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
            hidden_states, emb=temb
        )

    # Normalize encoder hidden states
    norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = (
        self.norm1_context(encoder_hidden_states, emb=temb)
    )

    # Normalize conditional latents if using them
    if use_cond:
        (
            norm_condition_latents,
            cond_gate_msa,
            cond_shift_mlp,
            cond_scale_mlp,
            cond_gate_mlp,
        ) = self.norm1(condition_latents, emb=cond_temb)

        (
            norm_condition_latents2,
            cond_gate_msa2,
            cond_shift_mlp2,
            cond_scale_mlp2,
            cond_gate_mlp2,
        ) = self.norm1(condition_latents2, emb=cond_temb)

    # Process through attention layer
    result = attn_forward(
        self.attn,
        model_config=model_config,
        hidden_states=norm_hidden_states,
        encoder_hidden_states=norm_encoder_hidden_states,
        condition_latents=norm_condition_latents if use_cond else None,
        condition_latents2=norm_condition_latents2 if use_cond else None,
        image_rotary_emb=image_rotary_emb,
        cond_rotary_emb=cond_rotary_emb if use_cond else None,
        cond_rotary_emb2=cond_rotary_emb2 if use_cond else None,
        latent_mask=latent_mask,
    )
    
    # Extract results from attention output
    attn_output, context_attn_output = result[:2]
    cond_attn_output = result[2] if use_cond else None
    cond_attn_output2 = result[3] if use_cond else None

    # Apply gating and residual connection to hidden states
    attn_output = gate_msa.unsqueeze(1) * attn_output
    hidden_states = hidden_states + attn_output
    
    # Apply gating and residual connection to encoder hidden states
    context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
    encoder_hidden_states = encoder_hidden_states + context_attn_output
    
    # Apply gating and residual connection to conditional latents
    if use_cond:
        # First condition
        cond_attn_output = cond_gate_msa.unsqueeze(1) * cond_attn_output
        condition_latents = condition_latents + cond_attn_output
        
        # Second condition
        cond_attn_output2 = cond_gate_msa2.unsqueeze(1) * cond_attn_output2
        condition_latents2 = condition_latents2 + cond_attn_output2
        
        # Optionally add conditional attention outputs to main hidden states
        if model_config.get("add_cond_attn", False):
            hidden_states += cond_attn_output
            hidden_states += cond_attn_output2

    # LayerNorm + MLP for hidden states
    norm_hidden_states = self.norm2(hidden_states)
    norm_hidden_states = (
        norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
    )
    
    # LayerNorm + MLP for encoder hidden states
    norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
    norm_encoder_hidden_states = (
        norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
    )
    
    # LayerNorm + MLP for conditional latents
    if use_cond:
        # First condition
        norm_condition_latents = self.norm2(condition_latents)
        norm_condition_latents = (
            norm_condition_latents * (1 + cond_scale_mlp[:, None])
            + cond_shift_mlp[:, None]
        )
        
        # Second condition
        norm_condition_latents2 = self.norm2(condition_latents2)
        norm_condition_latents2 = (
            norm_condition_latents2 * (1 + cond_scale_mlp2[:, None])
            + cond_shift_mlp2[:, None]
        )

    # Feed-forward network for hidden states
    with enable_lora((self.ff.net[2],), model_config.get("latent_lora", False)):
        ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output
    
    # Feed-forward network for encoder hidden states
    context_ff_output = self.ff_context(norm_encoder_hidden_states)
    context_ff_output = c_gate_mlp.unsqueeze(1) * context_ff_output
    
    # Feed-forward network for conditional latents
    if use_cond:
        # First condition
        with enable_lora((self.ff.net[2],), True):
            cond_ff_output = self.ff(norm_condition_latents)
            cond_ff_output = cond_gate_mlp.unsqueeze(1) * cond_ff_output
        
        # Second condition
        with enable_lora((self.ff.net[2],), True):
            cond_ff_output2 = self.ff(norm_condition_latents2)
            cond_ff_output2 = cond_gate_mlp2.unsqueeze(1) * cond_ff_output2

    # Apply residual connections
    hidden_states = hidden_states + ff_output
    encoder_hidden_states = encoder_hidden_states + context_ff_output
    
    if use_cond:
        condition_latents = condition_latents + cond_ff_output
        condition_latents2 = condition_latents2 + cond_ff_output2

    # Clip to avoid overflow in float16
    if encoder_hidden_states.dtype == torch.float16:
        encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

    return (
        encoder_hidden_states, 
        hidden_states, 
        condition_latents if use_cond else None, 
        condition_latents2 if use_cond else None
    )


def single_block_forward(
    self,
    hidden_states: torch.FloatTensor,
    temb: torch.FloatTensor,
    condition_latents: Optional[torch.FloatTensor] = None,
    condition_latents2: Optional[torch.FloatTensor] = None,
    cond_temb: Optional[torch.FloatTensor] = None,
    image_rotary_emb: Optional[torch.Tensor] = None,
    cond_rotary_emb: Optional[torch.Tensor] = None,
    cond_rotary_emb2: Optional[torch.Tensor] = None,
    model_config: Optional[Dict[str, Any]] = None,
    latent_mask: Optional[torch.Tensor] = None,
) -> Union[torch.FloatTensor, Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]]:
    """
    Forward pass for a single transformer block with optional conditional inputs.
    
    This implementation handles a simplified transformer block architecture where
    projection and normalization are applied before attention and MLP paths are
    merged at the end.
    
    Args:
        self: Block module instance
        hidden_states: Input hidden states
        temb: Timestep embeddings
        condition_latents: First conditional latent input
        condition_latents2: Second conditional latent input
        cond_temb: Timestep embeddings for conditional latents
        image_rotary_emb: Rotary embeddings for image
        cond_rotary_emb: Rotary embeddings for first condition
        cond_rotary_emb2: Rotary embeddings for second condition
        model_config: Model configuration dictionary
        latent_mask: Mask for mixing latent conditions
        
    Returns:
        Either processed hidden_states alone (if no conditions),
        or a tuple of (hidden_states, condition_latents, condition_latents2)
    """
    if model_config is None:
        model_config = {}
        
    using_cond = condition_latents is not None
    residual = hidden_states
    
    # Normalize hidden states and apply MLP projection
    with enable_lora(
        (
            self.norm.linear,
            self.proj_mlp,
        ),
        model_config.get("latent_lora", False),
    ):
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))
    
    # Process conditional latents if provided
    if using_cond:
        # Store residuals for each condition
        residual_cond = condition_latents
        residual_cond2 = condition_latents2
        
        # Process first condition
        with enable_lora(
            (
                self.norm.linear,
                self.proj_mlp,
            ),
            True,
        ):
            norm_condition_latents, cond_gate = self.norm(condition_latents, emb=cond_temb)
            mlp_cond_hidden_states = self.act_mlp(self.proj_mlp(norm_condition_latents))
        
        # Process second condition
        with enable_lora(
            (
                self.norm.linear,
                self.proj_mlp,
            ),
            True,
        ):
            norm_condition_latents2, cond_gate2 = self.norm(condition_latents2, emb=cond_temb)
            mlp_cond_hidden_states2 = self.act_mlp(self.proj_mlp(norm_condition_latents2))

    # Process through attention layer
    attn_output = attn_forward(
        self.attn,
        model_config=model_config,
        hidden_states=norm_hidden_states,
        image_rotary_emb=image_rotary_emb,
        latent_mask=latent_mask,
        **(
            {
                "condition_latents": norm_condition_latents,
                "condition_latents2": norm_condition_latents2,
                "cond_rotary_emb": cond_rotary_emb,
                "cond_rotary_emb2": cond_rotary_emb2,
            }
            if using_cond
            else {}
        ),
    )
    
    # Extract outputs for conditioned case
    if using_cond:
        attn_output, cond_attn_output, cond_attn_output2 = attn_output
    
    # Final projection and residual connection for main hidden states
    with enable_lora((self.proj_out,), model_config.get("latent_lora", False)):
        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
        gate = gate.unsqueeze(1)
        hidden_states = gate * self.proj_out(hidden_states)
        hidden_states = residual + hidden_states
    
    # Final projection and residual connection for conditional latents
    if using_cond:
        # First condition
        with enable_lora((self.proj_out,), True):
            condition_latents = torch.cat([cond_attn_output, mlp_cond_hidden_states], dim=2)
            cond_gate = cond_gate.unsqueeze(1)
            condition_latents = cond_gate * self.proj_out(condition_latents)
            condition_latents = residual_cond + condition_latents
        
        # Second condition
        with enable_lora((self.proj_out,), True):
            condition_latents2 = torch.cat([cond_attn_output2, mlp_cond_hidden_states2], dim=2)
            cond_gate2 = cond_gate2.unsqueeze(1)
            condition_latents2 = cond_gate2 * self.proj_out(condition_latents2)
            condition_latents2 = residual_cond2 + condition_latents2

    # Clip values for float16 to avoid overflow
    if hidden_states.dtype == torch.float16:
        hidden_states = hidden_states.clip(-65504, 65504)

    # Return appropriate outputs based on whether conditions were used
    return (
        (hidden_states, condition_latents, condition_latents2) 
        if using_cond 
        else hidden_states
    )