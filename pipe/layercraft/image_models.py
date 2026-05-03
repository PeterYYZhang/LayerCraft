"""Image model wrappers for FLUX and OpenAI image APIs."""

from __future__ import annotations

import base64
import gc
import io
import os
from typing import Any

from PIL import Image, ImageChops, ImageDraw


FLUX_BG_MODEL = "black-forest-labs/FLUX.2-klein-9B"
FLUX_FILL_MODEL = "black-forest-labs/FLUX.1-Fill-dev"
FLUX_BG_INFERENCE_STEPS = 4
FLUX_BG_GUIDANCE_SCALE = 1.0
GPT_IMAGE_MODEL = "gpt-image-2"
MIN_MASK_AREA = 75 * 75

_active_flux: dict[str, Any] = {"kind": None, "gpu_id": None, "pipe": None}


def validate_cuda_device(gpu_id: int) -> None:
    import torch

    count = torch.cuda.device_count()
    if gpu_id < 0 or gpu_id >= count:
        raise ValueError(f"GPU id {gpu_id} is unavailable; detected {count} CUDA devices")


def _torch_dtype() -> Any:
    import torch

    return torch.bfloat16 if torch.cuda.is_available() else torch.float32


def _hf_token() -> str:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required for gated FLUX model access")
    return token


def _clear_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def unload_flux_pipeline() -> None:
    """Move the active FLUX pipeline off GPU and drop it before loading another."""

    pipe = _active_flux.get("pipe")
    if pipe is not None:
        try:
            pipe.to("cpu")
        except Exception:
            pass
    _active_flux.update({"kind": None, "gpu_id": None, "pipe": None})
    gc.collect()
    _clear_cuda_cache()


def _load_exclusive_flux_pipeline(
    *,
    kind: str,
    gpu_id: int,
    pipeline_cls: Any,
    model_id: str,
) -> Any:
    if _active_flux["kind"] == kind and _active_flux["gpu_id"] == gpu_id:
        return _active_flux["pipe"]

    unload_flux_pipeline()
    validate_cuda_device(gpu_id)
    pipe = pipeline_cls.from_pretrained(
        model_id,
        token=_hf_token(),
        torch_dtype=_torch_dtype(),
    )
    pipe.to(f"cuda:{gpu_id}")
    _active_flux.update({"kind": kind, "gpu_id": gpu_id, "pipe": pipe})
    return pipe


def _flux_background_pipeline_cls() -> Any:
    try:
        from diffusers import Flux2KleinPipeline

        return Flux2KleinPipeline
    except ImportError:
        pass

    try:
        from diffusers import DiffusionPipeline
    except ImportError as exc:
        raise ImportError(
            "Flux2KleinPipeline is unavailable. Install a Diffusers release with "
            "FLUX.2 klein support using `pip install -U diffusers transformers "
            "accelerate`."
        ) from exc
    return DiffusionPipeline


def _flux_fill_pipeline_cls() -> Any:
    try:
        from diffusers import FluxFillPipeline

        return FluxFillPipeline
    except ImportError:
        pass

    try:
        from diffusers import FluxInpaintPipeline

        return FluxInpaintPipeline
    except ImportError as exc:
        raise ImportError(
            "Neither FluxFillPipeline nor FluxInpaintPipeline is available. "
            "Install a FLUX inpainting capable Diffusers release, for example "
            "`pip install -U diffusers transformers accelerate`."
        ) from exc


def get_flux_background_pipeline(gpu_id: int) -> Any:
    return _load_exclusive_flux_pipeline(
        kind="background",
        gpu_id=gpu_id,
        pipeline_cls=_flux_background_pipeline_cls(),
        model_id=FLUX_BG_MODEL,
    )


def get_flux_fill_pipeline(gpu_id: int) -> Any:
    return _load_exclusive_flux_pipeline(
        kind="fill",
        gpu_id=gpu_id,
        pipeline_cls=_flux_fill_pipeline_cls(),
        model_id=FLUX_FILL_MODEL,
    )


def make_bbox_mask(size: tuple[int, int], bbox: list[int] | tuple[int, int, int, int]) -> Image.Image:
    """Create a black background, white-filled bbox mask."""

    x, y, width, height = [int(value) for value in bbox]
    area = width * height
    if area <= MIN_MASK_AREA:
        raise ValueError(
            f"Mask bbox area must be greater than {MIN_MASK_AREA} pixels, got {area}"
        )
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle((x, y, x + width - 1, y + height - 1), fill=255)
    return mask


def make_bbox_boundary_mask(
    size: tuple[int, int],
    bboxes: list[list[int]] | list[tuple[int, int, int, int]],
    *,
    band: int = 24,
) -> Image.Image:
    """Create a white ring around each bbox boundary for final seam cleanup."""

    mask = Image.new("L", size, 0)
    canvas_width, canvas_height = size
    for bbox in bboxes:
        x, y, width, height = [int(value) for value in bbox]
        if width <= 0 or height <= 0:
            continue
        ring = Image.new("L", size, 0)
        draw = ImageDraw.Draw(ring)
        outer = (
            max(0, x - band),
            max(0, y - band),
            min(canvas_width - 1, x + width + band - 1),
            min(canvas_height - 1, y + height + band - 1),
        )
        draw.rectangle(outer, fill=255)
        inner = (
            max(0, x + band),
            max(0, y + band),
            min(canvas_width - 1, x + width - band - 1),
            min(canvas_height - 1, y + height - band - 1),
        )
        if inner[0] <= inner[2] and inner[1] <= inner[3]:
            draw.rectangle(inner, fill=0)
        mask = ImageChops.lighter(mask, ring)
    return mask


def composite_masked_edit(
    canvas: Image.Image,
    bbox_mask: Image.Image,
    edited: Image.Image,
) -> Image.Image:
    """Keep the original canvas outside the white mask area."""

    return Image.composite(
        edited.convert("RGB"),
        canvas.convert("RGB"),
        bbox_mask.convert("L"),
    )


def flux_generate_background(
    description: str,
    viewpoint: str,
    size: tuple[int, int],
    *,
    gpu_id: int,
) -> Image.Image:
    pipe = get_flux_background_pipeline(gpu_id)
    prompt = (
        f"{description}\n\nCamera/viewpoint: {viewpoint}\n"
        "No foreground subject objects; generate only the background environment."
    )
    result = pipe(
        prompt=prompt,
        width=size[0],
        height=size[1],
        guidance_scale=FLUX_BG_GUIDANCE_SCALE,
        num_inference_steps=FLUX_BG_INFERENCE_STEPS,
    )
    return result.images[0].convert("RGB")


def flux_generate_scene(
    description: str,
    viewpoint: str,
    size: tuple[int, int],
    *,
    gpu_id: int,
) -> Image.Image:
    pipe = get_flux_background_pipeline(gpu_id)
    prompt = f"{description}\n\nCamera/viewpoint: {viewpoint}"
    result = pipe(
        prompt=prompt,
        width=size[0],
        height=size[1],
        guidance_scale=FLUX_BG_GUIDANCE_SCALE,
        num_inference_steps=FLUX_BG_INFERENCE_STEPS,
    )
    return result.images[0].convert("RGB")


def flux_inpaint(
    canvas: Image.Image,
    bbox_mask: Image.Image,
    prompt: str,
    *,
    gpu_id: int,
) -> Image.Image:
    pipe = get_flux_fill_pipeline(gpu_id)
    result = pipe(
        prompt=prompt,
        image=canvas.convert("RGB"),
        mask_image=bbox_mask.convert("L"),
        width=canvas.width,
        height=canvas.height,
    )
    return composite_masked_edit(canvas, bbox_mask, result.images[0])


def _default_openai_client() -> Any:
    from openai import OpenAI

    return OpenAI()


def _decode_image(b64_json: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64_json))).convert("RGB")


def _png_file(image: Image.Image, name: str) -> io.BytesIO:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    buffer.name = name
    return buffer


def _openai_edit_mask(mask: Image.Image) -> Image.Image:
    """Convert white=inpaint mask to OpenAI transparent=edit RGBA mask."""

    grayscale = mask.convert("L")
    alpha = grayscale.point(lambda value: 0 if value > 127 else 255)
    rgba = Image.new("RGBA", grayscale.size, (0, 0, 0, 255))
    rgba.putalpha(alpha)
    return rgba


def gpt_image_generate(
    prompt: str,
    size: str = "1024x1024",
    *,
    client: Any | None = None,
) -> Image.Image:
    client = client or _default_openai_client()
    response = client.images.generate(model=GPT_IMAGE_MODEL, prompt=prompt, size=size)
    return _decode_image(response.data[0].b64_json)


def gpt_image_edit(
    canvas: Image.Image,
    bbox_mask: Image.Image,
    reference: Image.Image,
    prompt: str,
    *,
    size: str | None = None,
    client: Any | None = None,
) -> Image.Image:
    client = client or _default_openai_client()
    kwargs: dict[str, Any] = {
        "model": GPT_IMAGE_MODEL,
        "image": [
            _png_file(canvas.convert("RGB"), "canvas.png"),
            _png_file(reference.convert("RGB"), "reference.png"),
        ],
        "mask": _png_file(_openai_edit_mask(bbox_mask), "mask.png"),
        "prompt": prompt,
    }
    if size is not None:
        kwargs["size"] = size
    response = client.images.edit(**kwargs)
    return composite_masked_edit(canvas, bbox_mask, _decode_image(response.data[0].b64_json))


def gpt_image_cleanup(
    canvas: Image.Image,
    cleanup_mask: Image.Image,
    *,
    prompt: str,
    size: str | None = None,
    client: Any | None = None,
) -> Image.Image:
    client = client or _default_openai_client()
    kwargs: dict[str, Any] = {
        "model": GPT_IMAGE_MODEL,
        "image": _png_file(canvas.convert("RGB"), "canvas.png"),
        "mask": _png_file(_openai_edit_mask(cleanup_mask), "cleanup_mask.png"),
        "prompt": prompt,
    }
    if size is not None:
        kwargs["size"] = size
    response = client.images.edit(**kwargs)
    return composite_masked_edit(canvas, cleanup_mask, _decode_image(response.data[0].b64_json))


def gpt_image_polish(
    canvas: Image.Image,
    *,
    prompt: str,
    size: str | None = None,
    client: Any | None = None,
) -> Image.Image:
    client = client or _default_openai_client()
    kwargs: dict[str, Any] = {
        "model": GPT_IMAGE_MODEL,
        "image": _png_file(canvas.convert("RGB"), "canvas.png"),
        "prompt": prompt,
    }
    if size is not None:
        kwargs["size"] = size
    response = client.images.edit(**kwargs)
    return _decode_image(response.data[0].b64_json)
