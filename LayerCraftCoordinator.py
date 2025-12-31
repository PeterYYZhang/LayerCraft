"""
LayerCraft coordinator for layout planning and inpainting.
Runs ChainArchitect for planning and OIN for subject-driven edits.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from ChainArchitect import ChainArchitect

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class LayerCraftCoordinator:
    """Coordinates ChainArchitect for layout and OIN for inpainting."""

    def __init__(self, device: str = "cuda:0", output_dir: Optional[Path] = None):
        self.device = device
        self.output_dir = Path(output_dir) if output_dir else Path("outputs")
        self.output_dir.mkdir(exist_ok=True)
        self.architect = ChainArchitect(output_dir=self.output_dir)
        self._oin_module = None
        self._pipe = None

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _load_oin_module(self):
        """Import OIN-sdp.py despite the hyphen in its filename."""
        if self._oin_module is None:
            oin_path = Path(__file__).parent / "OIN-sdp.py"
            spec = importlib.util.spec_from_file_location("oin_sdp", oin_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Failed to load OIN module from {oin_path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)  # type: ignore[arg-type]
            self._oin_module = module
        return self._oin_module

    def _prepare_pipe(self, lora1: Optional[str], lora2: Optional[str]):
        """Initialize FLUX and load LoRAs when provided."""
        oin = self._load_oin_module()
        if self._pipe is None:
            logger.info("Loading FLUX.1 pipeline ...")
            self._pipe = oin.get_flux_pipe(device=self.device)
        if lora1 and lora2:
            logger.info("Loading LoRAs for fill/subject ...")
            self._pipe = oin.load_loras(self._pipe, lora1, lora2)
        return self._pipe

    def _maybe_generate_background(
        self, prompt: str, background_path: Optional[str], seed: int
    ) -> str:
        """Generate a background with FLUX when none is supplied."""
        if background_path:
            return background_path

        bg_path = self.output_dir / "background.png"
        logger.info("No background provided; generating with FLUX.1 ...")
        self._load_oin_module().generate_flux_only(
            prompt=prompt, save_path=str(bg_path), device=self.device, seed=seed
        )
        return str(bg_path)

    def _parse_plan(self, response: str) -> Dict[str, Any]:
        """Parse the planning JSON returned by ChainArchitect."""
        thinking, plan = self.architect._extract_json_parts(response)
        if "error" in plan:
            raise ValueError(plan["error"])
        return {"thinking": thinking, "plan": plan}

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def plan_scene(
        self, prompt: str, background_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Plan the scene with ChainArchitect and return the JSON."""
        if background_path:
            self.architect.add_image_message(prompt, [background_path])
            response = self.architect.get_response("")
        else:
            response = self.architect.get_response(prompt)
        return self._parse_plan(response)

    def run_pipeline(
        self,
        prompt: str,
        background_path: Optional[str],
        subject_map: Optional[Dict[str, str]],
        lora1: Optional[str],
        lora2: Optional[str],
        save_dir: Optional[str] = None,
        seed: int = 0,
        target_objects: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Plan with ChainArchitect, then integrate objects with OIN.

        Args:
            prompt: scene description.
            background_path: optional existing background.
            subject_map: object type -> subject image.
            lora1/lora2: optional LoRA checkpoints.
            save_dir: where to write results.
            seed: RNG seed for background generation.
            target_objects: optional subset of object types to place.
        Returns:
            Dict with the plan, intermediate renders, and final image path.
        """
        save_root = Path(save_dir) if save_dir else self.output_dir
        save_root.mkdir(exist_ok=True, parents=True)

        background_path = self._maybe_generate_background(prompt, background_path, seed)
        plan_bundle = self.plan_scene(prompt, background_path)
        plan = plan_bundle["plan"]

        placements: List[Dict[str, Any]] = plan.get("object_placements", [])
        if target_objects:
            target_set = {t.lower() for t in target_objects}
            placements = [
                p for p in placements if p.get("type", "").lower() in target_set
            ]

        placements = sorted(placements, key=lambda x: x.get("generation_order", 9999))

        if not placements:
            logger.warning("No object placements found; returning plan only.")
            return {
                "plan": plan,
                "thinking": plan_bundle["thinking"],
                "final": background_path,
            }

        oin = self._load_oin_module()
        pipe = self._prepare_pipe(lora1, lora2)

        current_bg = background_path
        intermediates = []
        for idx, obj in enumerate(placements, start=1):
            obj_type = obj.get("type", f"obj{idx}")
            bbox = [int(v) for v in obj.get("bounding_box", [])]
            obj_prompt = obj.get("prompt", obj_type)

            subject_path = None
            if subject_map:
                subject_path = subject_map.get(obj_type)
                if subject_path is None and len(subject_map) == 1:
                    subject_path = next(iter(subject_map.values()))

            if not subject_path:
                logger.warning("Skipping %s (no subject image provided).", obj_type)
                continue

            logger.info(
                "Integrating %s at %s bbox=%s ...",
                obj_type,
                obj.get("position"),
                bbox,
            )
            result = oin.generate_image(
                prompt=obj_prompt,
                bg_path=current_bg,
                subject_path=subject_path,
                bbox=bbox,
                pipe=pipe,
            )
            stage_path = save_root / f"stage_{idx:02d}_{obj_type}.png"
            oin.save_image(result, stage_path)
            intermediates.append(str(stage_path))
            current_bg = str(stage_path)

        return {
            "plan": plan,
            "thinking": plan_bundle["thinking"],
            "intermediates": intermediates,
            "final": current_bg,
        }


def parse_args():
    parser = argparse.ArgumentParser(
        description="LayerCraft Coordinator: plan with ChainArchitect and inpaint with OIN."
    )
    parser.add_argument("--prompt", required=True, help="User prompt / instruction.")
    parser.add_argument(
        "--background",
        type=str,
        default=None,
        help="Path to background image; if omitted, FLUX generates one.",
    )
    parser.add_argument(
        "--subject_map",
        type=str,
        default=None,
        help=(
            "JSON string mapping object type to subject image path "
            '(e.g., \'{"teddy bear":"bear.png"}\').'
        ),
    )
    parser.add_argument(
        "--subject",
        type=str,
        default=None,
        help="Single subject image to reuse for all objects when subject_map is not provided.",
    )
    parser.add_argument("--lora1", type=str, default=None, help="LoRA checkpoint for fill.")
    parser.add_argument("--lora2", type=str, default=None, help="LoRA checkpoint for subject.")
    parser.add_argument("--save_dir", type=str, default="outputs", help="Directory to save results.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device for FLUX/OIN.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for background generation.")
    parser.add_argument(
        "--targets",
        type=str,
        default=None,
        help="Comma separated list of object types to integrate (optional).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    subject_map = None
    if args.subject_map:
        subject_map = json.loads(args.subject_map)
    elif args.subject:
        subject_map = {"default": args.subject}

    targets = [t.strip() for t in args.targets.split(",")] if args.targets else None

    coordinator = LayerCraftCoordinator(device=args.device, output_dir=Path(args.save_dir))
    results = coordinator.run_pipeline(
        prompt=args.prompt,
        background_path=args.background,
        subject_map=subject_map,
        lora1=args.lora1,
        lora2=args.lora2,
        save_dir=args.save_dir,
        seed=args.seed,
        target_objects=targets,
    )

    logger.info("Final image saved to %s", results.get("final"))
    plan_path = Path(args.save_dir) / "chainarchitect_plan.json"
    plan_path.write_text(json.dumps(results.get("plan", {}), indent=2))
    logger.info("Plan written to %s", plan_path)

