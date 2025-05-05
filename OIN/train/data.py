"""
Conditioning Dataset Classes

This module provides dataset classes for training dual-condition diffusion models.
Each class prepares different types of conditioning inputs, such as edge maps,
depth maps, subjects with backgrounds, and other conditioning strategies.

The key classes include:
- Subject200KDateset: For paired image datasets with left/right images
- ImageConditionDataset: For various condition types like canny edges, depth, etc.
- LayerSubjectDataset: For layered images with subject, background and mask
"""

import random
import ast
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Union, Any

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image, ImageFilter, ImageDraw


def enlarge_bbox(bbox: List[int]) -> List[int]:
    """
    Enlarge a bounding box with specific padding ratios.
    
    Args:
        bbox: Original bounding box coordinates [x1, y1, x2, y2]
        
    Returns:
        Enlarged bounding box coordinates
    """
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    
    # Add padding: 5% on left and right, 15% on bottom
    bbox[0] = max(0, int(bbox[0] - width * 0.05))
    bbox[2] = min(512, int(bbox[2] + width * 0.05))
    bbox[3] = min(512, int(bbox[3] + height * 0.15))
    
    return bbox


class Subject200KDateset(Dataset):
    """
    Dataset for paired images (left/right) from the Subject200K dataset.
    
    This dataset handles paired images where one image serves as the target
    and the other as the condition. The pairs can be used in either order.
    
    Attributes:
        base_dataset: The source dataset containing paired images
        condition_size: Size of the condition image
        target_size: Size of the target image
        image_size: Base size of the original images
        padding: Padding around images
        condition_type: Type of conditioning to apply
        drop_text_prob: Probability of dropping text descriptions
        drop_image_prob: Probability of dropping condition images
        return_pil_image: Whether to return PIL images along with tensors
    """
    
    def __init__(
        self,
        base_dataset: Any,
        condition_size: int = 512,
        target_size: int = 512,
        image_size: int = 512,
        padding: int = 0,
        condition_type: str = "subject",
        drop_text_prob: float = 0.1,
        drop_image_prob: float = 0.1,
        return_pil_image: bool = False,
    ):
        """
        Initialize the Subject200K dataset.
        
        Args:
            base_dataset: Source dataset containing paired images
            condition_size: Size of the condition image (default: 512)
            target_size: Size of the target image (default: 512)
            image_size: Base size of the original images (default: 512)
            padding: Padding around images (default: 0)
            condition_type: Type of conditioning to apply (default: "subject")
            drop_text_prob: Probability of dropping text descriptions (default: 0.1)
            drop_image_prob: Probability of dropping condition images (default: 0.1)
            return_pil_image: Whether to return PIL images along with tensors (default: False)
        """
        self.base_dataset = base_dataset
        self.condition_size = condition_size
        self.target_size = target_size
        self.image_size = image_size
        self.padding = padding
        self.condition_type = condition_type
        self.drop_text_prob = drop_text_prob
        self.drop_image_prob = drop_image_prob
        self.return_pil_image = return_pil_image

        self.to_tensor = T.ToTensor()

    def __len__(self) -> int:
        """
        Return the length of the dataset (doubled since each pair can be used in two ways).
        
        Returns:
            Number of items in the dataset
        """
        return len(self.base_dataset) * 2

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get an item from the dataset.
        
        Args:
            idx: Index of the item to retrieve
            
        Returns:
            Dictionary containing:
            - image: Target image tensor
            - condition: Condition image tensor
            - condition_type: Type of conditioning
            - description: Text description (may be empty if dropped)
            - position_delta: Position adjustment for the condition
            - instance: Instance identifier
            - pil_image: Original PIL image (if return_pil_image is True)
        """
        # If target is 0, left image is target, right image is condition
        target = idx % 2
        item = self.base_dataset[idx // 2]

        # Crop the image to target and condition
        image = item["image"]
        left_img = image.crop(
            (
                self.padding,
                self.padding,
                self.image_size + self.padding,
                self.image_size + self.padding,
            )
        )
        right_img = image.crop(
            (
                self.image_size + self.padding * 2,
                self.padding,
                self.image_size * 2 + self.padding * 2,
                self.image_size + self.padding,
            )
        )

        # Get the target and condition image
        target_image, condition_img = (
            (left_img, right_img) if target == 0 else (right_img, left_img)
        )

        # Resize the images
        condition_img = condition_img.resize(
            (self.condition_size, self.condition_size)
        ).convert("RGB")
        target_image = target_image.resize(
            (self.target_size, self.target_size)
        ).convert("RGB")

        # Get the description
        description = item["description"][
            "description_0" if target == 0 else "description_1"
        ]
        
        # Randomly drop text or image
        drop_text = random.random() < self.drop_text_prob
        drop_image = random.random() < self.drop_image_prob
        if drop_text:
            description = ""
        if drop_image:
            condition_img = Image.new(
                "RGB", (self.condition_size, self.condition_size), (0, 0, 0)
            )

        return {
            "image": self.to_tensor(target_image),
            "condition": self.to_tensor(condition_img),
            "condition_type": self.condition_type,
            "description": description,
            # 16 is the downscale factor of the image
            "position_delta": np.array([0, -self.condition_size // 16]),
            "instance": item["description"]["item"],
            **({"pil_image": image} if self.return_pil_image else {}),
        }


class ImageConditionDataset(Dataset):
    """
    Dataset for various types of image conditioning.
    
    This dataset supports multiple conditioning types such as canny edges,
    depth maps, deblurring, colorization, inpainting, and super-resolution.
    
    Attributes:
        base_dataset: The source dataset containing images
        condition_size: Size of the condition image
        target_size: Size of the target image
        condition_type: Type of conditioning to apply
        drop_text_prob: Probability of dropping text descriptions
        drop_image_prob: Probability of dropping condition images
        return_pil_image: Whether to return PIL images along with tensors
    """
    
    def __init__(
        self,
        base_dataset: Any,
        condition_size: int = 512,
        target_size: int = 512,
        condition_type: str = "canny",
        drop_text_prob: float = 0.1,
        drop_image_prob: float = 0.1,
        return_pil_image: bool = False,
    ):
        """
        Initialize the image condition dataset.
        
        Args:
            base_dataset: Source dataset containing images
            condition_size: Size of the condition image (default: 512)
            target_size: Size of the target image (default: 512)
            condition_type: Type of conditioning to apply (default: "canny")
            drop_text_prob: Probability of dropping text descriptions (default: 0.1)
            drop_image_prob: Probability of dropping condition images (default: 0.1)
            return_pil_image: Whether to return PIL images along with tensors (default: False)
        """
        self.base_dataset = base_dataset
        self.condition_size = condition_size
        self.target_size = target_size
        self.condition_type = condition_type
        self.drop_text_prob = drop_text_prob
        self.drop_image_prob = drop_image_prob
        self.return_pil_image = return_pil_image

        self.to_tensor = T.ToTensor()
        self._depth_pipe = None

    def __len__(self) -> int:
        """
        Return the length of the dataset.
        
        Returns:
            Number of items in the dataset
        """
        return len(self.base_dataset)

    @property
    def depth_pipe(self):
        """
        Lazily initialize and return the depth estimation pipeline.
        
        Returns:
            Depth estimation pipeline from Hugging Face
        """
        if self._depth_pipe is None:
            from transformers import pipeline

            self._depth_pipe = pipeline(
                task="depth-estimation",
                model="LiheYoung/depth-anything-small-hf",
                device="cpu",
            )
        return self._depth_pipe

    def _get_canny_edge(self, img: Image.Image) -> Image.Image:
        """
        Generate Canny edge detection from an image.
        
        Args:
            img: Input image
            
        Returns:
            Edge-detected image
        """
        resize_ratio = self.condition_size / max(img.size)
        img = img.resize(
            (int(img.size[0] * resize_ratio), int(img.size[1] * resize_ratio))
        )
        img_np = np.array(img)
        img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(img_gray, 100, 200)
        return Image.fromarray(edges).convert("RGB")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get an item from the dataset with the specified conditioning.
        
        Args:
            idx: Index of the item to retrieve
            
        Returns:
            Dictionary containing:
            - image: Target image tensor
            - condition: Condition image tensor
            - condition_type: Type of conditioning
            - description: Text description (may be empty if dropped)
            - position_delta: Position adjustment for the condition
            - bbox: Bounding box for inpainting (if applicable)
            - pil_image: Original PIL images (if return_pil_image is True)
        """
        image = self.base_dataset[idx]["jpg"]
        image = image.resize((self.target_size, self.target_size)).convert("RGB")
        description = self.base_dataset[idx]["json"]["prompt"]
        bbox = None

        # Generate the condition image based on condition type
        position_delta = np.array([0, 0])
        if self.condition_type == "canny":
            # Edge detection conditioning
            condition_img = self._get_canny_edge(image)
            
        elif self.condition_type == "coloring":
            # Colorization conditioning (grayscale)
            condition_img = (
                image.resize((self.condition_size, self.condition_size))
                .convert("L")
                .convert("RGB")
            )
            
        elif self.condition_type == "deblurring":
            # Deblurring conditioning
            blur_radius = random.randint(1, 10)
            condition_img = (
                image.convert("RGB")
                .resize((self.condition_size, self.condition_size))
                .filter(ImageFilter.GaussianBlur(blur_radius))
                .convert("RGB")
            )
            
        elif self.condition_type == "depth":
            # Depth map conditioning
            condition_img = self.depth_pipe(image)["depth"].convert("RGB")
            
        elif self.condition_type == "depth_pred":
            # Depth prediction (swap condition and target)
            condition_img = image
            image = self.depth_pipe(condition_img)["depth"].convert("RGB")
            description = f"[depth] {description}"
            
        elif self.condition_type == "fill":
            # Inpainting conditioning
            condition_img = image.resize(
                (self.condition_size, self.condition_size)
            ).convert("RGB")
            w, h = image.size
            x1, x2 = sorted([random.randint(0, w), random.randint(0, w)])
            y1, y2 = sorted([random.randint(0, h), random.randint(0, h)])
            bbox = (x1, y1, x2, y2)
            mask = Image.new("L", image.size, 0)
            draw = ImageDraw.Draw(mask)
            draw.rectangle([x1, y1, x2, y2], fill=255)
            if random.random() > 0.5:
                mask = Image.eval(mask, lambda a: 255 - a)
            condition_img = Image.composite(
                image, Image.new("RGB", image.size, (0, 0, 0)), mask
            )
            
        elif self.condition_type == "sr":
            # Super-resolution conditioning
            condition_img = image.resize(
                (self.condition_size, self.condition_size)
            ).convert("RGB")
            position_delta = np.array([0, -self.condition_size // 16])

        else:
            raise ValueError(f"Condition type {self.condition_type} not implemented")

        # Randomly drop text or image
        drop_text = random.random() < self.drop_text_prob
        drop_image = random.random() < self.drop_image_prob
        if drop_text:
            description = ""
        if drop_image:
            condition_img = Image.new(
                "RGB", (self.condition_size, self.condition_size), (0, 0, 0)
            )

        return {
            "image": self.to_tensor(image),
            "condition": self.to_tensor(condition_img),
            "condition_type": self.condition_type,
            "description": description,
            "position_delta": position_delta,
            "bbox": bbox,
            **({"pil_image": [image, condition_img]} if self.return_pil_image else {}),
        }


class LayerSubjectDataset(Dataset):
    """
    Dataset for layered subject images with background and mask.
    
    This dataset is designed for the dual-condition training setup where one condition
    is the background with a masked-out subject, and the other condition is the subject.
    
    Attributes:
        base_dataset: The source dataset containing layered images
        condition_size: Size of the first condition image (background)
        condition_size2: Size of the second condition image (subject)
        target_size: Size of the target image
        image_size: Base size of the original images
        condition_type1: Type of first conditioning to apply (typically "fill")
        condition_type2: Type of second conditioning to apply (typically "subject")
        drop_text_prob: Probability of dropping text descriptions
        drop_image_prob: Probability of dropping condition images
        mask_bg: Whether to mask the background image
        return_pil_image: Whether to return PIL images along with tensors
    """
    
    def __init__(
        self,
        base_dataset: Any,
        condition_size: int = 512,
        condition_size2: int = 512,
        target_size: int = 512,
        image_size: int = 512,
        condition_type1: str = "fill",
        condition_type2: str = "subject",
        drop_text_prob: float = 0.05,
        drop_image_prob: float = 0.05,
        mask_bg: bool = True,
        return_pil_image: bool = False,
    ):
        """
        Initialize the layered subject dataset.
        
        Args:
            base_dataset: Source dataset containing layered images
            condition_size: Size of the first condition image (default: 512)
            condition_size2: Size of the second condition image (default: 512)
            target_size: Size of the target image (default: 512)
            image_size: Base size of the original images (default: 512)
            condition_type1: Type of first conditioning to apply (default: "fill")
            condition_type2: Type of second conditioning to apply (default: "subject")
            drop_text_prob: Probability of dropping text descriptions (default: 0.05)
            drop_image_prob: Probability of dropping condition images (default: 0.05)
            mask_bg: Whether to mask the background image (default: True)
            return_pil_image: Whether to return PIL images along with tensors (default: False)
        """
        self.base_dataset = base_dataset
        self.condition_size = condition_size
        self.condition_size2 = condition_size2
        self.target_size = target_size
        self.image_size = image_size
        self.condition_type1 = condition_type1
        self.condition_type2 = condition_type2
        self.drop_text_prob = drop_text_prob
        self.drop_image_prob = drop_image_prob
        self.mask_bg = mask_bg
        self.return_pil_image = return_pil_image

        self.to_tensor = T.ToTensor()

    def __len__(self) -> int:
        """
        Return the length of the dataset.
        
        Returns:
            Number of items in the dataset
        """
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get an item from the dataset with dual conditioning.
        
        Args:
            idx: Index of the item to retrieve
            
        Returns:
            Dictionary containing:
            - image: Target image tensor
            - condition: Subject image tensor (second condition)
            - bg: Background image tensor with masked subject (first condition)
            - origin_bg: Original background image tensor without masking
            - prompt: Combined instance and description text
            - bbox: Bounding box for the subject
            - instance: Instance identifier
            - description: Text description (may be empty if dropped)
            - condition_type1: Type of first conditioning
            - condition_type2: Type of second conditioning
            - position_delta1: Position adjustment for first condition
            - position_delta2: Position adjustment for second condition
            - pil_image: Original PIL images (if return_pil_image is True)
        """
        item = self.base_dataset[idx]

        # Extract images
        left_img = item["left_img"]       # Target image
        right_img = item["right_img"]     # Subject reference image
        bg = item["bg"]                   # Background image
        origin_bg = bg.copy()             # Original background (no masking)
        mask = item["mask"]               # Mask image
        instance = item["instance"]       # Instance name
        description = item["description"] # Description text

        # Resize all images to the specified size
        left_img = left_img.resize((self.image_size, self.image_size)).convert("RGB")
        right_img = right_img.resize((self.image_size, self.image_size)).convert("RGB")
        bg = bg.resize((self.image_size, self.image_size)).convert("RGB")
        mask = mask.resize((self.image_size, self.image_size)).convert("RGB")
        
        # Parse and process bounding box
        bbox = item["bbox"]
        if isinstance(bbox, str):
            bbox = ast.literal_eval(bbox)
        bbox = enlarge_bbox(bbox.copy())
        
        # Mask the background if enabled
        if self.mask_bg:
            bg.paste((0, 0, 0), (bbox[0], bbox[1], bbox[2], bbox[3]))
        
        # Randomly drop text or image
        drop_text = random.random() < self.drop_text_prob
        drop_image = random.random() < self.drop_image_prob
        
        if drop_text:
            description = ""
            
        if drop_image:
            right_img = Image.new(
                "RGB", (self.condition_size, self.condition_size), (0, 0, 0)
            )
            
        # Combine instance and description into prompt
        prompt = f"{instance} {description}"
        
        return {
            "image": self.to_tensor(left_img),
            "condition": self.to_tensor(right_img),
            "bg": self.to_tensor(bg),
            "origin_bg": self.to_tensor(origin_bg),
            "prompt": prompt,
            "bbox": bbox,
            "instance": instance,
            "description": description,
            "condition_type1": self.condition_type1,
            "condition_type2": self.condition_type2,
            "position_delta1": np.array([0, 0]),
            "position_delta2": np.array([0, -self.condition_size // 16]),
            **({"pil_image": [left_img, right_img, bg, mask]} if self.return_pil_image else {}),
        }