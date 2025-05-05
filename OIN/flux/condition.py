"""
Condition Input Processing Module

This module provides functionality for preparing and processing various types of 
conditional inputs for the Flux diffusion pipeline. It supports multiple condition
types such as depth maps, edge detection, subject preservation, and more.

Each condition type is processed differently to extract the relevant information
that guides the generation process. The Condition class handles the conversion of 
raw images to the appropriate format and provides methods to encode these conditions
for use in the model's forward pass.
"""

import torch
import numpy as np
import cv2
from typing import Optional, Union, List, Tuple, Dict, Any
from diffusers.pipelines import FluxPipeline
from PIL import Image, ImageFilter

from .pipeline_tools import encode_images


# Mapping from condition type names to their numeric IDs
# These IDs are used to distinguish different conditioning inputs in the model
condition_dict: Dict[str, int] = {
    "depth": 0,          # Depth map conditioning
    "canny": 1,          # Edge detection conditioning
    "subject": 4,        # Subject preservation
    "coloring": 6,       # Colorization (from grayscale)
    "deblurring": 7,     # Image deblurring
    "depth_pred": 8,     # Depth prediction
    "fill": 9,           # Inpainting/filling
    "sr": 10,            # Super-resolution
    "subject_bg": 11,    # Subject background modification
    "subject_obj": 12,   # Subject object modification
}


def enlarge_bbox(bbox: List[int]) -> List[int]:
    """
    Enlarge a bounding box with padding proportional to its dimensions.
    
    This helps ensure the subject is fully captured with some context around it.
    
    Args:
        bbox: Original bounding box coordinates [x1, y1, x2, y2]
        
    Returns:
        Enlarged bounding box coordinates
    """
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    
    # Add padding: 5% on left, 10% on right, 20% on bottom
    bbox[0] = max(0, int(bbox[0] - width * 0.05))
    bbox[2] = min(512, int(bbox[2] + width * 0.10))
    bbox[3] = min(512, int(bbox[3] + height * 0.20))
    
    return bbox


class Condition:
    """
    Class for preparing and processing different types of conditional inputs.
    
    This class handles the conversion of raw images into the appropriate format
    based on the condition type, and provides methods to encode these conditions
    for use in the Flux diffusion pipeline.
    
    Attributes:
        condition_type: Type of condition (e.g., "depth", "canny", "subject")
        condition: Processed condition image
        position_delta: Optional position adjustment for condition tokens
    """
    
    def __init__(
        self,
        condition_type: str,
        raw_img: Optional[Union[Image.Image, torch.Tensor]] = None,
        condition: Optional[Union[Image.Image, torch.Tensor]] = None,
        mask: Optional[Union[Image.Image, torch.Tensor]] = None,
        position_delta: Optional[List[int]] = None,
        bbox: Optional[List[int]] = None,
    ) -> None:
        """
        Initialize a Condition object.
        
        Args:
            condition_type: Type of condition to apply
            raw_img: Raw input image to process into a condition
            condition: Pre-processed condition (if raw_img is None)
            mask: Optional mask for the condition (not currently supported)
            position_delta: Optional position adjustment for condition tokens
            bbox: Optional bounding box for subject-related conditions
        
        Raises:
            AssertionError: If neither raw_img nor condition is provided
                           or if mask is provided (not yet supported)
        """
        self.condition_type = condition_type
        self.bbox = bbox
        
        assert raw_img is not None or condition is not None, "Either raw_img or condition must be provided"
        
        if raw_img is not None:
            self.condition = self.get_condition(condition_type, raw_img, bbox)
        else:
            self.condition = condition
            
        self.position_delta = position_delta
        
        # Mask support will be added in future versions
        assert mask is None, "Mask not supported yet"

    def get_condition(
        self, 
        condition_type: str, 
        raw_img: Union[Image.Image, torch.Tensor],
        bbox: Optional[List[int]] = None
    ) -> Union[Image.Image, torch.Tensor]:
        """
        Process raw image into the appropriate condition format.
        
        Different condition types require different processing methods:
        - depth: Generate depth map using depth-estimation model
        - canny: Apply Canny edge detection
        - subject: Preserve the original image
        - coloring: Convert to grayscale for colorization
        - deblurring: Apply Gaussian blur
        - fill: Use as-is for inpainting
        - subject_bg: Mask out the subject using bbox
        - subject_obj: Use as-is for subject modification
        
        Args:
            condition_type: Type of condition to generate
            raw_img: Raw input image
            bbox: Optional bounding box for subject-related conditions
            
        Returns:
            Processed condition image
            
        Raises:
            NotImplementedError: If condition type is not supported
        """
        if condition_type == "depth":
            # Generate depth map using depth-estimation model
            from transformers import pipeline

            depth_pipe = pipeline(
                task="depth-estimation",
                model="LiheYoung/depth-anything-small-hf",
                device="cuda",
            )
            source_image = raw_img.convert("RGB")
            condition_img = depth_pipe(source_image)["depth"].convert("RGB")
            return condition_img
            
        elif condition_type == "canny":
            # Apply Canny edge detection
            img = np.array(raw_img)
            edges = cv2.Canny(img, 100, 200)
            edges = Image.fromarray(edges).convert("RGB")
            return edges
            
        elif condition_type == "subject":
            # Use as-is for subject preservation
            return raw_img
            
        elif condition_type == "coloring":
            # Convert to grayscale for colorization
            return raw_img.convert("L").convert("RGB")
            
        elif condition_type == "deblurring":
            # Apply Gaussian blur to simulate a blurred image
            condition_image = (
                raw_img.convert("RGB")
                .filter(ImageFilter.GaussianBlur(10))
                .convert("RGB")
            )
            return condition_image
            
        elif condition_type == "fill":
            # Use as-is for inpainting/filling
            return raw_img.convert("RGB")
            
        elif condition_type == "subject_bg":
            # Create an image with the subject masked out
            assert bbox is not None, "bbox must be provided for subject_bg"
            bbox = enlarge_bbox(bbox.copy()) if bbox else None
            img_copy = raw_img.copy()
            img_copy.paste((0, 0, 0), (bbox[0], bbox[1], bbox[2], bbox[3]))
            return img_copy.convert("RGB")
            
        elif condition_type == "subject_obj":
            # Use as-is for subject object modification
            return raw_img.convert("RGB")

        # Return unmodified if condition type not recognized
        return self.condition

    @property
    def type_id(self) -> int:
        """
        Get the numeric ID for this condition type.
        
        Returns:
            Integer ID corresponding to the condition type
        """
        return condition_dict[self.condition_type]

    @classmethod
    def get_type_id(cls, condition_type: str) -> int:
        """
        Class method to get the numeric ID for any condition type.
        
        Args:
            condition_type: Name of the condition type
            
        Returns:
            Integer ID corresponding to the condition type
        """
        return condition_dict[condition_type]

    def encode(self, pipe: FluxPipeline) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode the condition for use in the Flux pipeline.
        
        This method converts the condition image into tokens and position IDs
        that can be used by the transformer model. It also generates the
        type ID tensor that identifies the condition type.
        
        Args:
            pipe: FluxPipeline instance to use for encoding
            
        Returns:
            Tuple of (tokens, position_ids, type_id)
            
        Raises:
            NotImplementedError: If encoding for the condition type is not implemented
        """
        # Check if condition type is supported for encoding
        if self.condition_type in [
            "depth",
            "canny",
            "subject",
            "coloring",
            "deblurring",
            "depth_pred",
            "fill",
            "sr",
            "subject_bg",
            "subject_obj",
        ]:
            # Encode image to tokens and position IDs
            tokens, ids = encode_images(pipe, self.condition)
        else:
            raise NotImplementedError(
                f"Condition type {self.condition_type} not implemented"
            )
        
        # Apply position adjustments for subject-related conditions if not explicitly provided
        if self.position_delta is None and (self.condition_type == "subject" or self.condition_type == "subject_obj"):
            self.position_delta = [0, -self.condition.size[0] // 16]
            
        # Apply position adjustments if specified
        if self.position_delta is not None:
            ids[:, 1] += self.position_delta[0]
            ids[:, 2] += self.position_delta[1]
            
        # Create type ID tensor matching the shape of position IDs
        type_id = torch.ones_like(ids[:, :1]) * self.type_id
        
        return tokens, ids, type_id