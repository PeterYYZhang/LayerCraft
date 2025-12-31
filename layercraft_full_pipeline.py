"""
Standalone runner for the LayerCraft pipeline.
Plans the scene, prepares a background, then integrates objects with OIN.
"""

import argparse
import importlib.util
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image

from ChainArchitect import ChainArchitect

# Load OIN-sdp (hyphenated filename) without renaming it.
_OIN_PATH = Path(__file__).parent / "OIN-sdp.py"
_oin_spec = importlib.util.spec_from_file_location("OIN_sdp", _OIN_PATH)
if _oin_spec is None or _oin_spec.loader is None:
    raise ImportError(f"Could not load OIN-sdp.py from {_OIN_PATH}")
OIN_sdp = importlib.util.module_from_spec(_oin_spec)
_oin_spec.loader.exec_module(OIN_sdp)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("layercraft_runner")


def _safe_prompt_dir(prompt: str, output_dir: Path) -> Path:
    safe = "".join(c if c.isalnum() else "_" for c in prompt[:50])
    path = output_dir / safe
    path.mkdir(parents=True, exist_ok=True)
    return path


def _load_subject_map(path: Optional[str]) -> Dict[str, str]:
    if not path:
        return {}
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("subject_map must be a JSON object mapping name/type -> image_path")
    return {k: str(v) for k, v in data.items()}


def _prep_bbox(box) -> Optional[List[int]]:
    if box is None:
        return None
    if isinstance(box, list) and len(box) == 4:
        return [int(x) for x in box]
    if isinstance(box, str):
        parts = box.split(",")
        if len(parts) == 4:
            return [int(x) for x in parts]
    return None


def run_pipeline(
    prompt: str,
    background_prompt: Optional[str],
    background_path: Optional[str],
    subject_map_path: Optional[str],
    lora1: Optional[str],
    lora2: Optional[str],
    device: str,
    output_dir: Path,
    width: int,
    height: int,
    image_paths: Optional[List[str]] = None,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_dir = _safe_prompt_dir(prompt, output_dir)

    # Plan layout with ChainArchitect
    architect = ChainArchitect(output_dir=prompt_dir)
    plan = architect.plan_layout(prompt, image_paths=image_paths)
    plan_json = plan.get("json_output", {})
    if "error" in plan_json:
        raise RuntimeError(f"ChainArchitect error: {plan_json['error']}")

    object_placements = plan_json.get("object_placements", [])
    if not object_placements:
        raise RuntimeError("No object_placements returned by ChainArchitect.")

    # Save plan JSON alongside outputs (already saved by architect; keep alias)
    plan_path = prompt_dir / "chainarchitect_object_placements.json"
    with open(plan_path, "w") as f:
        json.dump(plan_json, f, indent=2)

    # Prepare OIN pipeline
    pipe = OIN_sdp.get_flux_pipe(device)
    if lora1 and lora2:
        pipe = OIN_sdp.load_loras(pipe, lora1, lora2)
    else:
        logger.warning("LoRA paths not provided; using base FLUX for inpainting.")

    subjects = _load_subject_map(subject_map_path)

    # Background
    if background_path:
        current_image = Image.open(background_path).convert("RGB").resize((width, height))
        bg_path = Path(background_path)
        logger.info("Using provided background: %s", bg_path)
    else:
        bg_prompt = background_prompt or plan_json.get("background_analysis", {}).get("description", prompt)
        bg_path = prompt_dir / "background.png"
        logger.info("Generating background with FLUX. Prompt: %s", bg_prompt)
        current_image = OIN_sdp.generate_flux_only(
            prompt=bg_prompt,
            save_path=str(bg_path),
            device=device,
            height=height,
            width=width,
        )

    # Sequential object integration (farther first)
    placements_sorted = sorted(
        object_placements,
        key=lambda o: o.get("generation_order", sys.maxsize),
    )

    steps = []
    for idx, obj in enumerate(placements_sorted, start=1):
        obj_name = obj.get("type") or obj.get("name") or f"object_{idx}"
        obj_prompt = obj.get("prompt") or f"Integrate {obj_name} into the scene."
        bbox = _prep_bbox(obj.get("bounding_box"))
        if not bbox:
            logger.warning("Skipping %s: missing or invalid bbox.", obj_name)
            continue

        subject_path = subjects.get(obj_name)
        if subject_path is None:
            logger.warning("Skipping %s: subject image not provided in subject_map.", obj_name)
            continue

        logger.info("Inpainting %s with bbox %s", obj_name, bbox)
        current_image = OIN_sdp.generate_image(
            obj_prompt,
            bg_path=None,
            subject_path=subject_path,
            bbox=bbox,
            pipe=pipe,
            bg_image=current_image,
            subject_image=Image.open(subject_path),
        )

        step_path = prompt_dir / f"step_{idx:02d}_{obj_name}.png"
        OIN_sdp.save_image(current_image, step_path)
        steps.append({"object": obj_name, "path": step_path})

    final_path = steps[-1]["path"] if steps else bg_path
    logger.info("Pipeline complete. Final image: %s", final_path)

    return {
        "plan_path": plan_path,
        "background_path": bg_path,
        "steps": steps,
        "final_image_path": final_path,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Run LayerCraft pipeline without modifying existing files.")
    parser.add_argument("--prompt", required=True, type=str, help="User prompt to plan and render.")
    parser.add_argument("--background_prompt", type=str, default=None, help="Optional override for background text prompt.")
    parser.add_argument("--background_path", type=str, default=None, help="Existing background image to start from.")
    parser.add_argument("--subject_map", type=str, default=None, help="JSON mapping object name/type -> subject image path.")
    parser.add_argument("--lora1", type=str, default=None, help="Path to LoRA for fill adapter.")
    parser.add_argument("--lora2", type=str, default=None, help="Path to LoRA for subject adapter.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device for inference.")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to store outputs.")
    parser.add_argument("--width", type=int, default=512, help="Canvas width (OIN expects 512 bbox space).")
    parser.add_argument("--height", type=int, default=512, help="Canvas height (OIN expects 512 bbox space).")
    parser.add_argument("--images", type=str, nargs="*", default=None, help="Optional reference images for planning.")
    return parser.parse_args()


def main():
    args = parse_args()
    results = run_pipeline(
        prompt=args.prompt,
        background_prompt=args.background_prompt,
        background_path=args.background_path,
        subject_map_path=args.subject_map,
        lora1=args.lora1,
        lora2=args.lora2,
        device=args.device,
        output_dir=Path(args.output_dir),
        width=args.width,
        height=args.height,
        image_paths=args.images,
    )
    logger.info("Final outputs: %s", results)


if __name__ == "__main__":
    main()

