"""Coordinator agent and real tool handlers."""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from . import image_models
from .architect import plan_layout as architect_plan_layout
from .canvas import CanvasRegistry, draw_bbox_overlay
from .llm import DEFAULT_LLM_MODEL, run_agent
from .recorder import RunRecorder, safe_name
from .tools import coordinator_tools


SYSTEM_PROMPT = (
    "You are the LayerCraft Coordinator. Use the tools to plan, generate, validate, inpaint, and finish.\n"
    "Follow each tool description and keep retries within the stated budgets.\n"
    "After layout planning, use generate_full_scene for simple scenes that can be rendered as one coherent image.\n"
    "After layout planning, process objects strictly in layout order: validate one object's bbox overlay, "
    "update that object's bbox position or size if needed, inpaint it, validate that rendered object, "
    "regenerate with bbox size/position overrides if needed, then move on."
)


VisionValidator = Callable[[Image.Image, dict[str, Any]], dict[str, Any]]


def _data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _response_content(response: Any) -> str:
    message = response.choices[0].message
    content = getattr(message, "content", "")
    if isinstance(content, list):
        return "\n".join(str(part.get("text", part)) for part in content)
    return str(content)


def _vision_json(
    client: Any,
    *,
    image: Image.Image,
    layout: dict[str, Any],
    instruction: str,
    model: str,
) -> dict[str, Any]:
    response = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": "Return strict JSON with keys ok:boolean and issues:array.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {
                        "type": "text",
                        "text": "Planned layout JSON:\n" + json.dumps(layout),
                    },
                    {"type": "image_url", "image_url": {"url": _data_url(image)}},
                ],
            },
        ],
    )
    content = _response_content(response)
    parsed = json.loads(content)
    return {"ok": bool(parsed.get("ok", False)), "issues": parsed.get("issues", [])}


class Coordinator:
    """Runs the simplified LayerCraft inference flow."""

    def __init__(
        self,
        *,
        gpu_id: int,
        canvas_size: tuple[int, int] = (1024, 1024),
        client: Any | None = None,
        model: str = DEFAULT_LLM_MODEL,
        runs_root: Path | None = None,
        output_root: Path | None = None,
        agent_runner: Callable[..., Any] | None = None,
        layout_validator: VisionValidator | None = None,
        canvas_validator: VisionValidator | None = None,
    ) -> None:
        self.gpu_id = gpu_id
        self.canvas_size = canvas_size
        self.client = client
        self.model = model
        self.runs_root = runs_root
        self.output_root = output_root or Path.cwd()
        self.agent_runner = agent_runner or run_agent
        self.layout_validator = layout_validator
        self.canvas_validator = canvas_validator
        self.final_path: Path | None = None

    def generate(self, prompt: str) -> Image.Image:
        recorder = RunRecorder(prompt, runs_root=self.runs_root)
        registry = CanvasRegistry(recorder=recorder)
        state: dict[str, Any] = {
            "layout": None,
            "layout_version": 0,
            "layout_validation_count": 0,
            "canvas_validation_count": 0,
            "layout_refinements": 0,
            "current_image_id": None,
            "references": {},
            "modes": {},
            "retries": {},
            "rendered_objects": [],
            "layout_checked_objects": [],
            "accepted_objects": [],
            "object_base_image_ids": {},
            "full_scene_generated": False,
        }

        def require_layout() -> dict[str, Any]:
            layout = state["layout"]
            if not isinstance(layout, dict):
                raise RuntimeError("No layout has been planned yet")
            return layout

        def resolve_current_canvas(image_id: str, tool_name: str) -> tuple[str, Image.Image]:
            current_image_id = state["current_image_id"]
            if current_image_id is None:
                raise RuntimeError(f"{tool_name} requires a current canvas")
            return current_image_id, registry.get(current_image_id)

        def find_object(name: str) -> tuple[int, dict[str, Any]]:
            layout = require_layout()
            for index, obj in enumerate(layout["objects"], start=1):
                if obj["name"] == name:
                    return index, obj
            raise KeyError(f"Unknown object: {name}")

        def save_layout() -> None:
            recorder.save_json(f"layout_v{state['layout_version']}.json", require_layout())

        def update_object_bbox(object_name: str, bbox: list[int]) -> None:
            _, obj = find_object(object_name)
            normalized_bbox = [int(value) for value in bbox]
            if obj.get("bbox") == normalized_bbox:
                return
            obj["bbox"] = normalized_bbox
            state["layout_version"] += 1
            save_layout()

        def plan_layout(args: dict[str, Any]) -> dict[str, Any]:
            layout = architect_plan_layout(
                args["prompt"],
                self.canvas_size,
                client=self.client,
                model=self.model,
                recorder=recorder,
            )
            state["layout"] = layout
            state["layout_version"] = 0
            save_layout()
            return layout

        def generate_background(args: dict[str, Any]) -> dict[str, Any]:
            size = tuple(args.get("size") or self.canvas_size)
            image = image_models.flux_generate_background(
                args["description"],
                args["viewpoint"],
                (int(size[0]), int(size[1])),
                gpu_id=self.gpu_id,
            )
            image_id = registry.register(image, artifact_name="background.png")
            state["current_image_id"] = image_id
            return {"image_id": image_id}

        def simple_scene_issues(layout: dict[str, Any]) -> list[str]:
            objects = layout.get("objects", [])
            if len(objects) > 3:
                return ["more than three planned objects"]
            if any(obj.get("requires_reference") for obj in objects):
                return ["one or more objects require a reference image"]
            relation_count = sum(len(obj.get("relations", [])) for obj in objects)
            if relation_count > 5:
                return ["too many spatial/object relations for one-shot generation"]
            return []

        def full_scene_prompt(layout: dict[str, Any]) -> str:
            background = layout["background"]
            object_lines = [
                f"- {obj['name']}: {obj['description']} ({'; '.join(obj.get('relations', []))})"
                for obj in layout["objects"]
            ]
            description = (
                background["description"]
                + "\n\nInclude these foreground objects directly in the generated scene:\n"
                + "\n".join(object_lines)
                + "\n\nGenerate the complete scene in one pass with natural composition."
            )
            return description

        def generate_full_scene(args: dict[str, Any]) -> dict[str, Any]:
            layout = require_layout()
            issues = simple_scene_issues(layout)
            if issues:
                return {
                    "skipped": True,
                    "issues": ["Scene is not simple enough for one-shot generation", *issues],
                }
            size = tuple(args.get("size") or layout.get("canvas_size") or self.canvas_size)
            image = image_models.flux_generate_scene(
                full_scene_prompt(layout),
                layout["background"]["viewpoint"],
                (int(size[0]), int(size[1])),
                gpu_id=self.gpu_id,
            )
            image_id = registry.register(image, artifact_name="full_scene.png")
            state["current_image_id"] = image_id
            state["full_scene_generated"] = True
            for obj in layout["objects"]:
                mark_object_accepted(obj["name"])
            return {"image_id": image_id, "mode": "full_scene"}

        def object_layout(obj: dict[str, Any]) -> dict[str, Any]:
            layout = require_layout()
            return {
                "canvas_size": layout.get("canvas_size", list(self.canvas_size)),
                "background": layout.get("background", {}),
                "objects": [obj],
            }

        def validate_layout(args: dict[str, Any]) -> dict[str, Any]:
            pending = current_unaccepted_object_name()
            if pending is not None:
                return {
                    "ok": False,
                    "skipped": True,
                    "expected_object": pending,
                    "issues": [
                        f"Validate or regenerate rendered object {pending} before checking the next layout"
                    ],
                }
            object_name = args.get("object_name") or next_layout_object_name()
            if object_name is None:
                return {"ok": True, "issues": []}
            expected = next_layout_object_name()
            if expected is not None and object_name != expected:
                return {
                    "ok": False,
                    "skipped": True,
                    "expected_object": expected,
                    "issues": [f"Validate layout for {expected} before {object_name}"],
                }
            _, obj = find_object(object_name)
            _, canvas = resolve_current_canvas(
                args["background_image_id"],
                "validate_layout",
            )
            layout = object_layout(obj)
            overlay = draw_bbox_overlay(canvas, layout)
            index = state["layout_validation_count"]
            recorder.save_image(
                f"overlay-{index}",
                overlay,
                artifact_name=f"overlay_v{index}_{safe_name(object_name)}.png",
            )
            validator = self.layout_validator or self._validate_layout_with_vision
            verdict = validator(overlay, layout)
            recorder.save_json(f"validate_layout_v{index}.json", verdict)
            state["layout_validation_count"] = index + 1
            if verdict.get("ok"):
                mark_layout_checked(object_name)
            elif state["layout_refinements"] >= 2:
                verdict["exhausted"] = True
                verdict.setdefault("issues", []).append(
                    f"Layout refinement budget exhausted for {object_name}; continuing with current bbox"
                )
                mark_layout_checked(object_name)
            return verdict

        def update_layout(args: dict[str, Any]) -> dict[str, Any]:
            if state["layout_refinements"] >= 2:
                return {
                    "ok": False,
                    "issues": ["layout refinement budget exhausted"],
                    "layout": require_layout(),
                }
            pending = current_unaccepted_object_name()
            if pending is not None:
                return {
                    "ok": False,
                    "skipped": True,
                    "expected_object": pending,
                    "issues": [
                        f"Validate or regenerate rendered object {pending} before updating the next layout"
                    ],
                    "layout": require_layout(),
                }
            expected = next_layout_object_name()
            adjustments = args["adjustments"]
            if expected is not None:
                if len(adjustments) != 1 or adjustments[0]["object_name"] != expected:
                    return {
                        "ok": False,
                        "skipped": True,
                        "expected_object": expected,
                        "issues": [
                            f"Update only the current layout object {expected} before rendering"
                        ],
                        "layout": require_layout(),
                    }
            layout = require_layout()
            by_name = {obj["name"]: obj for obj in layout["objects"]}
            for adjustment in adjustments:
                obj = by_name[adjustment["object_name"]]
                for key in ("bbox", "description", "order", "requires_reference"):
                    if key in adjustment:
                        obj[key] = adjustment[key]
            layout["objects"].sort(key=lambda item: int(item["order"]))
            if expected in state["layout_checked_objects"]:
                state["layout_checked_objects"].remove(expected)
            state["layout_refinements"] += 1
            state["layout_version"] += 1
            save_layout()
            return layout

        def object_artifact(
            object_name: str,
            mode: str,
            *,
            regen: int | None = None,
        ) -> str:
            index, _ = find_object(object_name)
            suffix = f"_regen{regen}" if regen is not None else ""
            return f"canvas_step{index:02d}_{safe_name(object_name)}_{mode}{suffix}.png"

        def mask_artifact(object_name: str, *, regen: int | None = None) -> str:
            index, _ = find_object(object_name)
            suffix = f"_regen{regen}" if regen is not None else ""
            return f"mask_step{index:02d}_{safe_name(object_name)}{suffix}.png"

        def next_layout_object_name() -> str | None:
            accepted = set(state["accepted_objects"])
            if any(object_name not in accepted for object_name in state["rendered_objects"]):
                return None
            rendered = set(state["rendered_objects"])
            for obj in require_layout()["objects"]:
                if obj["name"] not in rendered:
                    return obj["name"]
            return None

        def mark_layout_checked(object_name: str) -> None:
            checked = state["layout_checked_objects"]
            if object_name not in checked:
                checked.append(object_name)

        def next_object_name() -> str | None:
            accepted = set(state["accepted_objects"])
            for obj in require_layout()["objects"]:
                if obj["name"] not in accepted:
                    return obj["name"]
            return None

        def skip_if_not_next_object(object_name: str) -> dict[str, Any] | None:
            current_image_id = state["current_image_id"]
            if current_image_id is None:
                return None
            rendered = state["rendered_objects"]
            accepted = state["accepted_objects"]
            if object_name in accepted:
                return {
                    "image_id": current_image_id,
                    "skipped": True,
                    "issues": [f"{object_name} has already been accepted"],
                }
            expected = next_object_name()
            if expected is not None and object_name != expected:
                return {
                    "image_id": current_image_id,
                    "skipped": True,
                    "expected_object": expected,
                    "issues": [f"Render {expected} before {object_name}"],
                }
            if object_name in rendered:
                return {
                    "image_id": current_image_id,
                    "skipped": True,
                    "issues": [
                        f"{object_name} has already been rendered; validate or regenerate it"
                    ],
                }
            if object_name not in state["layout_checked_objects"]:
                return {
                    "image_id": current_image_id,
                    "skipped": True,
                    "expected_object": object_name,
                    "issues": [
                        f"Validate and approve the layout bbox for {object_name} before inpainting"
                    ],
                }
            return None

        def mark_object_rendered(object_name: str) -> None:
            rendered = state["rendered_objects"]
            if object_name not in rendered:
                rendered.append(object_name)

        def remember_object_base(object_name: str, image_id: str) -> None:
            base_ids = state["object_base_image_ids"]
            if object_name not in base_ids:
                base_ids[object_name] = image_id

        def object_base_canvas(object_name: str) -> tuple[str, Image.Image]:
            base_id = state["object_base_image_ids"].get(object_name)
            if not isinstance(base_id, str):
                raise RuntimeError(f"No base canvas recorded for {object_name}")
            return base_id, registry.get(base_id)

        def current_unaccepted_object_name() -> str | None:
            accepted = set(state["accepted_objects"])
            for object_name in reversed(state["rendered_objects"]):
                if object_name not in accepted:
                    return object_name
            return None

        def mark_object_accepted(object_name: str) -> None:
            accepted = state["accepted_objects"]
            if object_name not in accepted:
                accepted.append(object_name)

        def inpaint_with_text(args: dict[str, Any]) -> dict[str, Any]:
            skip = skip_if_not_next_object(args["object_name"])
            if skip is not None:
                return skip
            current_image_id, canvas = resolve_current_canvas(args["image_id"], "inpaint_with_text")
            remember_object_base(args["object_name"], current_image_id)
            mask = image_models.make_bbox_mask(canvas.size, args["bbox"])
            recorder.save_image(
                f"mask-{args['object_name']}",
                mask,
                artifact_name=mask_artifact(args["object_name"]),
            )
            image = image_models.flux_inpaint(
                canvas,
                mask,
                args["description"],
                gpu_id=self.gpu_id,
            )
            image_id = registry.register(
                image,
                artifact_name=object_artifact(args["object_name"], "text"),
            )
            state["current_image_id"] = image_id
            state["modes"][args["object_name"]] = "text"
            mark_object_rendered(args["object_name"])
            return {"image_id": image_id}

        def create_reference(args: dict[str, Any]) -> dict[str, Any]:
            prompt_text = (
                "Clean isolated reference image on a plain neutral background: "
                + args["description"]
            )
            image = image_models.gpt_image_generate(
                prompt_text,
                size=args.get("size", "1024x1024"),
                client=self.client,
            )
            image_id = registry.register(
                image,
                artifact_name=f"reference_{safe_name(args['object_name'])}.png",
            )
            state["references"][args["object_name"]] = image_id
            return {"image_id": image_id}

        def inpaint_with_reference(args: dict[str, Any]) -> dict[str, Any]:
            skip = skip_if_not_next_object(args["object_name"])
            if skip is not None:
                return skip
            current_image_id, canvas = resolve_current_canvas(args["image_id"], "inpaint_with_reference")
            remember_object_base(args["object_name"], current_image_id)
            reference = registry.get(args["reference_image_id"])
            mask = image_models.make_bbox_mask(canvas.size, args["bbox"])
            recorder.save_image(
                f"mask-{args['object_name']}",
                mask,
                artifact_name=mask_artifact(args["object_name"]),
            )
            image = image_models.gpt_image_edit(
                canvas,
                mask,
                reference,
                args["description"],
                size=f"{canvas.width}x{canvas.height}",
                client=self.client,
            )
            image_id = registry.register(
                image,
                artifact_name=object_artifact(args["object_name"], "ref"),
            )
            state["current_image_id"] = image_id
            state["modes"][args["object_name"]] = "reference"
            mark_object_rendered(args["object_name"])
            return {"image_id": image_id}

        def validate_canvas(args: dict[str, Any]) -> dict[str, Any]:
            object_name = args.get("object_name") or current_unaccepted_object_name()
            if object_name is None:
                return {"ok": True, "issues": []}
            current = current_unaccepted_object_name()
            if current is not None and object_name != current:
                current_image_id = state["current_image_id"]
                return {
                    "ok": False,
                    "image_id": current_image_id,
                    "skipped": True,
                    "expected_object": current,
                    "issues": [f"Validate {current} before {object_name}"],
                }
            _, obj = find_object(object_name)
            _, canvas = resolve_current_canvas(args["image_id"], "validate_canvas")
            validator = self.canvas_validator or self._validate_canvas_with_vision
            verdict = validator(canvas, object_layout(obj))
            index = state["canvas_validation_count"]
            recorder.save_json(f"validate_canvas_v{index}.json", verdict)
            state["canvas_validation_count"] = index + 1
            if verdict.get("ok"):
                mark_object_accepted(object_name)
            elif int(state["retries"].get(object_name, 0)) >= 2:
                verdict["exhausted"] = True
                verdict.setdefault("issues", []).append(
                    f"Retry budget exhausted for {object_name}; continuing to next object"
                )
                mark_object_accepted(object_name)
            return verdict

        def regenerate_object(args: dict[str, Any]) -> dict[str, Any]:
            object_name = args["object_name"]
            current_image_id, _ = resolve_current_canvas(args["image_id"], "regenerate_object")
            current = current_unaccepted_object_name()
            if current is not None and object_name != current:
                return {
                    "image_id": current_image_id,
                    "skipped": True,
                    "expected_object": current,
                    "issues": [f"Regenerate {current} before {object_name}"],
                }
            retry_count = int(state["retries"].get(object_name, 0))
            if retry_count >= 2:
                mark_object_accepted(object_name)
                return {
                    "image_id": current_image_id,
                    "exhausted": True,
                    "retries": retry_count,
                }
            _, obj = find_object(object_name)
            bbox = args.get("override_bbox") or obj["bbox"]
            if "override_bbox" in args:
                update_object_bbox(object_name, bbox)
            description = args.get("override_description") or obj["description"]
            _, canvas = object_base_canvas(object_name)
            mask = image_models.make_bbox_mask(canvas.size, bbox)
            regen_number = retry_count + 1
            recorder.save_image(
                f"mask-{object_name}-regen{regen_number}",
                mask,
                artifact_name=mask_artifact(object_name, regen=regen_number),
            )
            mode = args["mode"]
            if mode == "reference":
                reference_id = state["references"].get(object_name)
                if reference_id is None:
                    reference_result = create_reference(
                        {"object_name": object_name, "description": description}
                    )
                    reference_id = reference_result["image_id"]
                image = image_models.gpt_image_edit(
                    canvas,
                    mask,
                    registry.get(reference_id),
                    description,
                    size=f"{canvas.width}x{canvas.height}",
                    client=self.client,
                )
                artifact_mode = "ref"
            else:
                image = image_models.flux_inpaint(
                    canvas,
                    mask,
                    description,
                    gpu_id=self.gpu_id,
                )
                artifact_mode = "text"
            image_id = registry.register(
                image,
                artifact_name=object_artifact(
                    object_name,
                    artifact_mode,
                    regen=regen_number,
                ),
            )
            state["retries"][object_name] = regen_number
            state["current_image_id"] = image_id
            state["modes"][object_name] = mode
            mark_object_rendered(object_name)
            return {"image_id": image_id, "exhausted": False, "retries": regen_number}

        def finish(args: dict[str, Any]) -> dict[str, Any]:
            current_image_id, image = resolve_current_canvas(args["image_id"], "finish")
            layout = require_layout()
            accepted = set(state["accepted_objects"])
            remaining = [
                obj["name"]
                for obj in layout["objects"]
                if obj["name"] not in accepted
            ]
            if remaining:
                expected = current_unaccepted_object_name() or remaining[0]
                return {
                    "image_id": current_image_id,
                    "skipped": True,
                    "expected_object": expected,
                    "remaining_objects": remaining,
                    "issues": [
                        "Cannot finish before every planned object is rendered and accepted"
                    ],
                }
            image = image_models.gpt_image_polish(
                image,
                prompt=(
                    "Subtly polish this final composed image. Improve local seams, "
                    "lighting consistency, perspective coherence, and small artifacts. "
                    "Preserve the exact scene layout, object identities, positions, "
                    "sizes, colors, and content; do not add, remove, or move objects."
                ),
                size=f"{image.width}x{image.height}",
                client=self.client,
            )
            current_image_id = registry.register(
                image,
                artifact_name="final_polish.png",
            )
            state["current_image_id"] = current_image_id
            self.final_path = recorder.copy_to_final(image, output_root=self.output_root)
            recorder.finalize(status="finished")
            return {"image_id": current_image_id, "final_path": str(self.final_path)}

        handlers = {
            "plan_layout": plan_layout,
            "generate_background": generate_background,
            "generate_full_scene": generate_full_scene,
            "validate_layout": validate_layout,
            "update_layout": update_layout,
            "inpaint_with_text": inpaint_with_text,
            "create_reference": create_reference,
            "inpaint_with_reference": inpaint_with_reference,
            "validate_canvas": validate_canvas,
            "regenerate_object": regenerate_object,
            "finish": finish,
        }
        try:
            result = self.agent_runner(
                SYSTEM_PROMPT,
                prompt,
                coordinator_tools(handlers),
                model=self.model,
                client=self.client,
                recorder=recorder,
                agent_name="Coordinator",
            )
            if not isinstance(result, dict) or "image_id" not in result:
                raise RuntimeError("Coordinator did not finish with an image_id")
            return registry.get(result["image_id"])
        except Exception:
            recorder.finalize(status="failed")
            raise

    def _validate_layout_with_vision(
        self,
        overlay: Image.Image,
        layout: dict[str, Any],
    ) -> dict[str, Any]:
        if self.client is None:
            from openai import OpenAI

            client = OpenAI()
        else:
            client = self.client
        return _vision_json(
            client,
            image=overlay,
            layout=layout,
            model=self.model,
            instruction=(
                "Check whether each labeled bbox has both sensible position and sensible size "
                "for this specific background. When not ok, suggest an adjusted bbox that may "
                "change x, y, width, and height."
            ),
        )

    def _validate_canvas_with_vision(
        self,
        canvas: Image.Image,
        layout: dict[str, Any],
    ) -> dict[str, Any]:
        if self.client is None:
            from openai import OpenAI

            client = OpenAI()
        else:
            client = self.client
        return _vision_json(
            client,
            image=canvas,
            layout=layout,
            model=self.model,
            instruction=(
                "Compare the rendered canvas against the planned layout. Flag missing "
                "objects, wrong positions, wrong bbox size/scale, identity drift, or style "
                "mismatch. When position or scale is wrong, suggest a corrected bbox with "
                "updated x, y, width, and height."
            ),
        )
