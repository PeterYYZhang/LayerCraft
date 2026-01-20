import os
import torch
from diffusers import FluxPipeline
from OIN.flux.condition import Condition
from PIL import Image
from OIN.flux.generate import generate

def get_flux_pipe(device= "cuda:0"):
    """
    Load the FluxPipeline model.
    """
    pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16
)
    pipe = pipe.to(device)
    return pipe




def load_loras(pipe, lora_path1, lora_path2):
    """
    Load the LoRA models into the pipeline.
    """
    weight_name = "pytorch_lora_weights.safetensors"

    # Resolve absolute paths to avoid HF Hub lookup when running from other CWDs
    lora_path1 = os.path.abspath(lora_path1)
    lora_path2 = os.path.abspath(lora_path2)

    lora1_file = os.path.join(lora_path1, weight_name)
    lora2_file = os.path.join(lora_path2, weight_name)

    if not os.path.isfile(lora1_file):
        raise FileNotFoundError(f"LoRA weight not found: {lora1_file}")
    if not os.path.isfile(lora2_file):
        raise FileNotFoundError(f"LoRA weight not found: {lora2_file}")

    pipe.load_lora_weights(lora_path1, weight_name=weight_name, adapter_name="fill")
    pipe.load_lora_weights(lora_path2, weight_name=weight_name, adapter_name="subject")
    return pipe


def generate_image(prompt, bg_path, subject_path, bbox, pipe):
    """
    Subject-driven inpainting using the OIN pipeline.
    """
    bg = Image.open(bg_path)
    subject = Image.open(subject_path)
    bg = bg.crop((0, 0, 512, 512))

    bg_copy = bg.copy()
    bg.paste((0,0,0), bbox)
    
    subject = subject.resize((512, 512))
    bg = Condition(
        condition_type="fill",
        condition=bg.resize((512, 512)).convert("RGB"),
        position_delta=[0, 0],
    )

    subject = Condition(
        condition_type="subject",
        condition=subject.resize((512, 512)).convert("RGB"),
        position_delta=[0, -32],
    )

    result = generate(
        pipe,
        prompt = prompt,
        conditions=[bg],
        conditions2=[subject],
        height=512,
        width=512,
        bbox=bbox,
        latents_temp=bg_copy.resize((512,512)).convert("RGB"),
        num_inference_steps=28,
        default_lora=True,
    ).images[0]

    return result

def save_image(image, save_path):
    """
    Save the generated image to the specified path.
    """
    # Check if save_path is a directory
    if os.path.isdir(save_path) or not any(save_path.endswith(ext) for ext in ['.png', '.jpg', '.jpeg']):
        # If it's a directory or has no valid extension, append a default filename with extension
        save_path = os.path.join(save_path, "generated_image.png")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    
    image.save(save_path)
    print(f"Image saved to {save_path}")

def argparse():
    """
    Parse command line arguments.
    """
    import argparse
    parser = argparse.ArgumentParser(description="Subject-driven inpainting using OIN.")
    parser.add_argument("--prompt", type=str, default="Fill the blank area in the given background with A small, golden-brown teddy bear with a smiling face and soft plush texture., with description: lying on the floor near the edge of the area rug, adding a touch of warmth and playfulness to the minimalist space.", help="Prompt for the generation.")
    parser.add_argument("--bg_path", type=str, default="/scratch3/ck1_23/OminiControl/examples/inpaint_6.png", help="Path to the background image.")
    parser.add_argument("--subject_path", type=str, default="/scratch3/ck1_23/OminiControl/examples/lora_1_cond_322 copy.jpg", help="Path to the subject image.")
    parser.add_argument("--bbox", type=str, default="300,300,500,490", help="Bounding box for inpainting (x1,y1,x2,y2).")
    parser.add_argument("--save_path", type=str, default="./", help="Path to save the generated image.")
    parser.add_argument("--pth1", type=str, default="/scratch3/ck1_23/OminiControl/mutil-test/20250303-170618/ckpt/15000_fill_fill/", help="Path to the first LoRA model.")
    parser.add_argument("--pth2", type=str, default="/scratch3/ck1_23/OminiControl/mutil-test/20250303-170618/ckpt/15000_subject_subject/", help="Path to the second LoRA model.")
    return parser.parse_args()

if __name__ == "__main__":
    args = argparse()
    device = "cuda:0"
    pipe = get_flux_pipe(device)
    lora_path1 = args.pth1
    lora_path2 = args.pth2
    pipe = load_loras(pipe, lora_path1, lora_path2)

    prompt = args.prompt
    bg_path = args.bg_path
    subject_path = args.subject_path
    bbox = [int(coord) for coord in args.bbox.split(",")]
    save_path = args.save_path

    result_image = generate_image(prompt, bg_path, subject_path, bbox, pipe)
    save_image(result_image, save_path)

# Usage:        
# python subject_driven_impainting.py --prompt "Fill the blank area in the given background with A small, golden-brown teddy bear with a smiling face and soft plush texture., with description: lying on the floor near the edge of the area rug, adding a touch of warmth and playfulness to the minimalist space." --bg_path /scratch3/ck1_23/OminiControl/examples/inpaint_6.png --subject_path /scratch3/ck1_23/OminiControl/examples/lora_1_cond_322 copy.jpg --bbox 300,300,500,490 --save_path ./
# python subject_driven_inpainting.py --prompt "Fill the blank area in the given background with A graduation bear, with description: holding by the guy." --bg_path "/scratch3/LayerCraft/IMG_1133 Medium.png" --subject_path /scratch3/LayerCraft/fluxtest_7.png --bbox 170,340,340,500