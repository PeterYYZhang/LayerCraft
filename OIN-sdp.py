import os
import torch
from diffusers import FluxPipeline
from OIN.flux.condition import Condition
from PIL import Image
from OIN.flux.generate import generate

def get_flux_pipe(device="cuda:0"):
    """Load the FLUX.1 diffusion pipeline."""
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16
    )
    pipe = pipe.to(device)
    return pipe




def load_loras(pipe, lora_path1, lora_path2):
    """Attach LoRA weights for the fill and subject adapters."""
    pipe.load_lora_weights(lora_path1, adapter_name="fill")
    pipe.load_lora_weights(lora_path2, adapter_name="subject")
    return pipe


def generate_image(prompt, bg_path, subject_path, bbox, pipe):
    """Subject-driven inpainting with the OIN pipeline."""
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
    """Save the generated image to disk."""
    # Check if save_path is a directory
    if os.path.isdir(save_path) or not any(save_path.endswith(ext) for ext in ['.png', '.jpg', '.jpeg']):
        # If it's a directory or has no valid extension, append a default filename with extension
        save_path = os.path.join(save_path, "generated_image.png")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    
    image.save(save_path)
    print(f"Image saved to {save_path}")

def argparse():
    """Parse command line arguments."""
    import argparse
    parser = argparse.ArgumentParser(description="Subject-driven inpainting using OIN.")
    parser.add_argument("--prompt", type=str, default="Fill the blank area with a small teddy bear near the rug edge.", help="Prompt for generation.")
    parser.add_argument("--bg_path", type=str, default="/scratch3/ck1_23/OminiControl/examples/inpaint_6.png", help="Path to the background image.")
    parser.add_argument("--subject_path", type=str, default="/scratch3/ck1_23/OminiControl/examples/lora_1_cond_322 copy.jpg", help="Path to the subject image.")
    parser.add_argument("--bbox", type=str, default="300,300,500,490", help="Bounding box for inpainting (x1,y1,x2,y2).")
    parser.add_argument("--save_path", type=str, default="./", help="Path to save the generated image.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device for generation.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for generation when LoRAs are not provided.")
    parser.add_argument("--pth1", type=str, default=None, help="Path to the first LoRA model.")
    parser.add_argument("--pth2", type=str, default=None, help="Path to the second LoRA model.")
    return parser.parse_args()

def generate_flux_only(prompt, save_path, device="cuda:0", height=1024, width=1024, guidance_scale=3.5, num_inference_steps=50, seed=0):
    """Plain FLUX.1 generation for cases without LoRA checkpoints."""
    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
    # Offload to CPU to reduce VRAM usage; remove if you prefer full GPU residency.
    pipe.enable_model_cpu_offload()
    generator_device = "cpu" if device is None else device
    generator = torch.Generator(generator_device).manual_seed(seed)
    image = pipe(
        prompt,
        height=height,
        width=width,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        max_sequence_length=512,
        generator=generator,
    ).images[0]
    save_image(image, save_path)
    return image

if __name__ == "__main__":
    args = argparse()
    # If LoRA paths are missing, fall back to plain FLUX generation with the given prompt.
    if not args.pth1 or not args.pth2:
        generate_flux_only(
            prompt=args.prompt,
            save_path=args.save_path,
            device=args.device,
            seed=args.seed,
        )
    else:
        pipe = get_flux_pipe(args.device)
        pipe = load_loras(pipe, args.pth1, args.pth2)

        prompt = args.prompt
        bg_path = args.bg_path
        subject_path = args.subject_path
        bbox = [int(coord) for coord in args.bbox.split(",")]
        save_path = args.save_path

        result_image = generate_image(prompt, bg_path, subject_path, bbox, pipe)
        save_image(result_image, save_path)

# Usage:        
# python OIN.py --prompt "Fill the blank area in the given background with A small, golden-brown teddy bear with a smiling face and soft plush texture., with description: lying on the floor near the edge of the area rug, adding a touch of warmth and playfulness to the minimalist space." --bg_path /scratch3/ck1_23/OminiControl/examples/inpaint_6.png --subject_path /scratch3/ck1_23/OminiControl/examples/lora_1_cond_322 copy.jpg --bbox 300,300,500,490 --save_path ./
# python OIN.py --prompt "Fill the blank area in the given background with A graduation bear, with description: holding by the guy." --bg_path "/scratch3/LayerCraft/IMG_1133 Medium.png" --subject_path /scratch3/LayerCraft/fluxtest_7.png --bbox 170,340,340,500