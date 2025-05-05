from diffusers.pipelines import FluxPipeline
from diffusers.utils import logging
from diffusers.pipelines.flux.pipeline_flux import logger
from torch import Tensor
import torch
import torch.nn.functional as F
from typing import Tuple, Optional
import math


def downsample_bbox_to_mask(latent, bbox: Optional[Tuple[int, int, int, int]], downsample_ratio=8)-> torch.Tensor:
    # mask = torch.zeros(latent.shape, device=latent.device)
    batch_size, c,  height, width = latent.shape
    mask = torch.zeros((batch_size, height, width), device=latent.device)
    for i in range(batch_size):
        if batch_size == 1:
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        else:
            x1, y1, x2, y2 = bbox[i][0], bbox[i][1], bbox[i][2], bbox[i][3]
        x1, y1, x2, y2 = x1 // downsample_ratio, y1 // downsample_ratio, x2 // downsample_ratio, y2 // downsample_ratio
        mask[i, y1:y2, x1:x2] = 1
    return mask.bool()

def pack_mask(mask: torch.Tensor) -> torch.Tensor:
    batch_size, height, width = mask.shape
    # Reshape to split into 2x2 blocks
    mask_blocks = mask.view(batch_size, height//2, 2, width//2, 2)
    # Aggregate: If any pixel in the 2x2 block is 1, mark the token as masked (1)
    mask_packed = mask_blocks.amax(dim=(2, 4))  # Max across 2x2 spatial blocks
    # Flatten to sequence: [batch, seq_len= (height//2 * width//2)]
    mask_packed = mask_packed.flatten(start_dim=1)
    return mask_packed

def encode_images(pipeline: FluxPipeline, images: Tensor, to_vae: bool = False):
    images = pipeline.image_processor.preprocess(images)
    images = images.to(pipeline.device).to(pipeline.dtype)
    images = pipeline.vae.encode(images).latent_dist.sample()
    images = (
        images - pipeline.vae.config.shift_factor
    ) * pipeline.vae.config.scaling_factor
    if to_vae:
        return images
    images_tokens = pipeline._pack_latents(images, *images.shape)
    images_ids = pipeline._prepare_latent_image_ids(
        images.shape[0],
        images.shape[2],
        images.shape[3],
        pipeline.device,
        pipeline.dtype,
    )
    if images_tokens.shape[1] != images_ids.shape[0]:
        images_ids = pipeline._prepare_latent_image_ids(
            images.shape[0],
            images.shape[2] // 2,
            images.shape[3] // 2,
            pipeline.device,
            pipeline.dtype,
        )
    return images_tokens, images_ids

def pack_latents(pipeline: FluxPipeline, images: Tensor):
    image_tokens = pipeline._pack_latents(images, *images.shape)
    image_ids = pipeline._prepare_latent_image_ids(
        images.shape[0],
        images.shape[2],
        images.shape[3],
        pipeline.device,
        pipeline.dtype,
    )
    if image_tokens.shape[1] != image_ids.shape[0]:
        image_ids = pipeline._prepare_latent_image_ids(
            images.shape[0],
            images.shape[2] // 2,
            images.shape[3] // 2,
            pipeline.device,
            pipeline.dtype,
        )
    return image_tokens, image_ids


def prepare_text_input(pipeline: FluxPipeline, prompts, max_sequence_length=512):
    # Turn off warnings (CLIP overflow)
    logger.setLevel(logging.ERROR)
    (
        prompt_embeds,
        pooled_prompt_embeds,
        text_ids,
    ) = pipeline.encode_prompt(
        prompt=prompts,
        prompt_2=None,
        prompt_embeds=None,
        pooled_prompt_embeds=None,
        device=pipeline.device,
        num_images_per_prompt=1,
        max_sequence_length=max_sequence_length,
        lora_scale=None,
    )
    # Turn on warnings
    logger.setLevel(logging.WARNING)
    return prompt_embeds, pooled_prompt_embeds, text_ids



def permute_by_mask(data: torch.Tensor, mask: torch.Tensor):
    """
    Permutes the tokens in `data` based on the boolean `mask` so that tokens with mask==False (unmasked)
    are placed before tokens with mask==True (masked) for each sample in the batch.
    
    Args:
        data (torch.Tensor): Tensor of shape (batch_size, seq_len, dim).
        mask (torch.Tensor): Boolean tensor of shape (batch_size, seq_len) where True indicates masked token.
    
    Returns:
        permuted_data (torch.Tensor): Tensor of shape (batch_size, seq_len, dim) with tokens permuted.
        perm_idx (torch.Tensor): Tensor of shape (batch_size, seq_len) containing the permutation indices.
    """
    if data.dim() == 3:
        batch_size, seq_len, dim = data.shape
        permuted_data = torch.empty_like(data)
        perm_idx = torch.empty((batch_size, seq_len), dtype=torch.long, device=data.device)
        
        for b in range(batch_size):
            # Convert mask to float so that False (0) comes before True (1) when sorting.
            # Ensure the mask for each sample is 1D
            current_mask = mask[b].view(-1)
            # Sort: False (0) will come before True (1)
            indices = torch.argsort(current_mask.float(), descending=False)
            # Ensure indices is 1D (squeeze any extra singleton dimension)
            indices = indices.squeeze()
            perm_idx[b] = indices
            permuted_data[b] = data[b, indices, :]
        
        return permuted_data, perm_idx
    elif data.dim() == 2: # for positional embeddings
        seq_len, dim = data.shape
        permuted_data = torch.empty_like(data)
        perm_idx = torch.empty((seq_len), dtype=torch.long, device=data.device)
        indices = torch.argsort(mask.float(), descending=False)
        perm_idx = indices
        permuted_data = data[indices]
        return permuted_data, perm_idx

def repermute_by_mask(permuted_data: torch.Tensor, perm_idx: torch.Tensor):
    """
    Reverts the permutation applied by permute_by_mask to recover the original token order.
    
    Args:
        permuted_data (torch.Tensor): Tensor of shape (batch_size, seq_len, dim) with permuted tokens.
        perm_idx (torch.Tensor): Tensor of shape (batch_size, seq_len) containing the permutation indices.
    
    Returns:
        original_data (torch.Tensor): Tensor of shape (batch_size, seq_len, dim) with the original order restored.
    """
    if permuted_data.dim() == 3:
        batch_size, seq_len, dim = permuted_data.shape
        original_data = torch.empty_like(permuted_data)
        
        for b in range(batch_size):
            # Compute inverse permutation.
            inv_perm = torch.empty_like(perm_idx[b])
            inv_perm[perm_idx[b]] = torch.arange(seq_len, device=perm_idx.device)
            original_data[b] = permuted_data[b, inv_perm, :]
        
        return original_data
    elif permuted_data.dim() == 2:
        batch_size, seq_len = permuted_data.shape
        original_data = torch.empty_like(permuted_data)

        for b in range(batch_size):
            inv_perm = torch.empty_like(perm_idx[b])
            inv_perm[perm_idx[b]] = torch.arange(seq_len, device=perm_idx.device)
            original_data[b] = permuted_data[b, inv_perm]

        return original_data

def partial_update(
    old_tensor: torch.FloatTensor,
    update: torch.FloatTensor,
    mask: Optional[torch.BoolTensor]
):
    """
    old_tensor:  [batch_size, seq_len, dim]
    update:      [batch_size, seq_len, dim]
    mask:        [batch_size, seq_len] or [batch_size, seq_len, 1]  (bool)
                 True means "update this position", 
                 False means "leave old_tensor unchanged".

    If mask is None, we fall back to a full update (just do old_tensor + update).
    """
    if mask is None:
        # Fallback: update everything
        return old_tensor + update

    # If mask is 2D, expand last dimension for broadcast
    if mask.dim() == 2:
        mask = mask.unsqueeze(-1)  # [B, L] => [B, L, 1]

    # Only add the update where mask == True
    return torch.where(mask, old_tensor + update, old_tensor)



# Example usage:
if __name__ == '__main__':
    # Create dummy data and mask.
    batch_size, seq_len, dim = 2, 10, 4
    data = torch.randn(batch_size, seq_len, dim)
    # Create a mask with different patterns for each sample.
    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    mask[0, 3:7] = True   # For sample 0, tokens 3 to 6 are masked.
    mask[1, 5:] = True    # For sample 1, tokens 5 to 9 are masked.

    permuted_data, perm_idx = permute_by_mask(data, mask)
    recovered_data = repermute_by_mask(permuted_data, perm_idx)

    # Check that the recovered data matches the original.
    print(torch.allclose(data, recovered_data))


def compute_similarity(tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:
    """
    Compute cosine similarity between two tensors.
    
    Args:
        tensor1: First input tensor
        tensor2: Second input tensor with same batch size as tensor1
        
    Returns:
        Tensor containing cosine similarity values for each item in the batch
    """
    tensor1_flat = tensor1.view(tensor1.size(0), -1)
    tensor2_flat = tensor2.view(tensor2.size(0), -1)
    similarity = F.cosine_similarity(tensor1_flat, tensor2_flat, dim=1)
    return similarity


def print_differs(tensor1: torch.Tensor, tensor2: torch.Tensor) -> None:
    """
    Print the mean absolute difference between two tensors.
    
    Args:
        tensor1: First input tensor
        tensor2: Second input tensor with same shape as tensor1
    """
    diff = (tensor1 - tensor2).abs().mean()
    print(f"Min diff: {diff.item()}")
