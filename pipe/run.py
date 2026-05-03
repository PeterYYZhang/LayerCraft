"""CLI entry point for simplified LayerCraft inference."""

from __future__ import annotations

import argparse
import os
import sys

from layercraft.coordinator import Coordinator
from layercraft.image_models import validate_cuda_device


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LayerCraft inference")
    parser.add_argument("--prompt", required=True, help="Text prompt to render")
    parser.add_argument("--gpu", required=True, type=int, help="CUDA device id")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY must be set")
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN must be set")
    validate_cuda_device(args.gpu)

    coordinator = Coordinator(gpu_id=args.gpu)
    coordinator.generate(args.prompt)
    if coordinator.final_path is None:
        raise RuntimeError("Coordinator finished without a final path")
    print(coordinator.final_path.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
