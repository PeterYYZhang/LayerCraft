"""Tool schemas and factories for the LayerCraft agents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping


ToolHandler = Callable[[dict[str, Any]], Any]


def _missing_handler(name: str) -> ToolHandler:
    def handler(_: dict[str, Any]) -> Any:
        raise NotImplementedError(f"No handler registered for tool '{name}'")

    return handler


@dataclass(frozen=True)
class Tool:
    """A callable tool exposed to an LLM tool-use loop."""

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: ToolHandler
    terminal: bool = False

    def validate(self) -> None:
        if not self.name:
            raise ValueError("Tool name is required")
        if not self.description:
            raise ValueError(f"Tool '{self.name}' needs a description")
        if self.input_schema.get("type") != "object":
            raise ValueError(f"Tool '{self.name}' schema must be an object")
        properties = self.input_schema.get("properties")
        if not isinstance(properties, dict):
            raise ValueError(f"Tool '{self.name}' schema needs properties")
        required = self.input_schema.get("required", [])
        if not isinstance(required, list):
            raise ValueError(f"Tool '{self.name}' required must be a list")
        missing = [key for key in required if key not in properties]
        if missing:
            raise ValueError(f"Tool '{self.name}' required keys missing: {missing}")

    def openai_schema(self) -> dict[str, Any]:
        self.validate()
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }


def _handler(handlers: Mapping[str, ToolHandler] | None, name: str) -> ToolHandler:
    if handlers and name in handlers:
        return handlers[name]
    return _missing_handler(name)


def _bbox_schema() -> dict[str, Any]:
    return {
        "type": "array",
        "description": (
            "Bounding box as [x, y, width, height] in pixel coordinates. "
            "The mask area width * height must be greater than 75 * 75 pixels."
        ),
        "items": {"type": "integer", "minimum": 0},
        "minItems": 4,
        "maxItems": 4,
    }


def architect_tools(
    handlers: Mapping[str, ToolHandler] | None = None,
) -> list[Tool]:
    """Tools used by ChainArchitect to build a structured scene layout."""

    return [
        Tool(
            name="set_background",
            description=(
                "Records the background scene and camera viewpoint. Call exactly once "
                "before adding objects. The background must describe only the static "
                "environment, not the foreground objects."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "Detailed background scene without foreground objects.",
                    },
                    "viewpoint": {
                        "type": "string",
                        "description": "Camera viewpoint, framing, lens feel, and perspective.",
                    },
                },
                "required": ["description", "viewpoint"],
                "additionalProperties": False,
            },
            handler=_handler(handlers, "set_background"),
        ),
        Tool(
            name="add_object",
            description=(
                "Registers one foreground object. order ascending = distant to near, "
                "so later inpainting can respect occlusion. Use relations such as "
                "'on top of sofa', 'left of lamp', or 'facing left'. Bboxes must fit "
                "inside the requested canvas."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Stable object name."},
                    "description": {
                        "type": "string",
                        "description": "Visual description used for inpainting.",
                    },
                    "bbox": _bbox_schema(),
                    "order": {
                        "type": "integer",
                        "description": "Distant-to-near occlusion order.",
                    },
                    "relations": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Spatial or semantic relations to other objects.",
                    },
                    "requires_reference": {
                        "type": "boolean",
                        "description": (
                            "True when identity, branding, character consistency, or "
                            "style consistency matters."
                        ),
                    },
                },
                "required": [
                    "name",
                    "description",
                    "bbox",
                    "order",
                    "relations",
                    "requires_reference",
                ],
                "additionalProperties": False,
            },
            handler=_handler(handlers, "add_object"),
        ),
        Tool(
            name="submit_layout",
            description=(
                "Terminates the loop and returns the accumulated layout. Call after "
                "the background is set and all foreground objects have been added."
            ),
            input_schema={
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
            handler=_handler(handlers, "submit_layout"),
            terminal=True,
        ),
    ]


def coordinator_tools(
    handlers: Mapping[str, ToolHandler] | None = None,
) -> list[Tool]:
    """Tools used by the Coordinator to run the full inference sequence."""

    return [
        Tool(
            name="plan_layout",
            description=(
                "Decomposes a scene into a structured layout with background plus "
                "ordered objects, distant first for occlusion. Use this once before "
                "generating any image."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "Original user prompt."}
                },
                "required": ["prompt"],
                "additionalProperties": False,
            },
            handler=_handler(handlers, "plan_layout"),
        ),
        Tool(
            name="generate_background",
            description=(
                "Creates a full canvas with no foreground objects. Always called "
                "before validating layout or inpainting."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "Background-only image prompt.",
                    },
                    "viewpoint": {
                        "type": "string",
                        "description": "Camera viewpoint from the layout.",
                    },
                    "size": {
                        "type": "array",
                        "items": {"type": "integer", "minimum": 1},
                        "minItems": 2,
                        "maxItems": 2,
                        "description": "Canvas size as [width, height].",
                    },
                },
                "required": ["description", "viewpoint", "size"],
                "additionalProperties": False,
            },
            handler=_handler(handlers, "generate_background"),
        ),
        Tool(
            name="generate_full_scene",
            description=(
                "Fast path for simple scenes. After plan_layout, call this when the "
                "whole prompt can be generated as one coherent image without separate "
                "bbox-controlled inpainting. The Coordinator checks whether the layout "
                "is simple enough; if the result is skipped, continue with "
                "generate_background and the normal object-by-object flow."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "size": {
                        "type": "array",
                        "items": {"type": "integer", "minimum": 1},
                        "minItems": 2,
                        "maxItems": 2,
                        "description": "Canvas size as [width, height].",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Why this planned layout is simple enough for one-shot generation.",
                    },
                },
                "required": ["size", "reason"],
                "additionalProperties": False,
            },
            handler=_handler(handlers, "generate_full_scene"),
        ),
        Tool(
            name="validate_layout",
            description=(
                "Pre-inpaint layout check for exactly one object at a time. Overlay "
                "the next object in layout order on the current canvas, ask GPT-5.4 "
                "vision whether that bbox has a sensible position and size, and update "
                "only that object's bbox x, y, width, or height if needed before "
                "inpainting it. Do not validate or update future objects in parallel."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "background_image_id": {
                        "type": "string",
                        "description": (
                            "Image id returned by generate_background or the latest "
                            "canvas id; the Coordinator will use the latest canvas."
                        ),
                    },
                    "object_name": {
                        "type": "string",
                        "description": (
                            "Optional current object name. Omit to validate the next "
                            "object in layout order."
                        ),
                    },
                },
                "required": ["background_image_id"],
                "additionalProperties": False,
            },
            handler=_handler(handlers, "validate_layout"),
        ),
        Tool(
            name="update_layout",
            description=(
                "Apply bbox, description, reference need, or order overrides to the "
                "current object only. Use this to act on validate_layout feedback "
                "including bbox size or position changes "
                "before inpainting that same object. Do not batch future-object "
                "layout changes."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "adjustments": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "object_name": {"type": "string"},
                                "bbox": _bbox_schema(),
                                "description": {"type": "string"},
                                "order": {"type": "integer"},
                                "requires_reference": {"type": "boolean"},
                            },
                            "required": ["object_name"],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["adjustments"],
                "additionalProperties": False,
            },
            handler=_handler(handlers, "update_layout"),
        ),
        Tool(
            name="inpaint_with_text",
            description=(
                "Text-only inpainting via FLUX.1-Fill-dev on the local GPU. Inputs "
                "are the latest canvas, black/white bbox mask, and text prompt. Use "
                "for generic objects where identity preservation is not required. "
                "Always pass the image_id returned by the immediately previous "
                "background, inpaint, or regenerate step. Render objects in the "
                "layout order only after validate_layout has approved that object's "
                "bbox; if the result says skipped with expected_object, handle that "
                "object next."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "image_id": {"type": "string"},
                    "bbox": _bbox_schema(),
                    "description": {"type": "string"},
                    "object_name": {"type": "string"},
                },
                "required": ["image_id", "bbox", "description", "object_name"],
                "additionalProperties": False,
            },
            handler=_handler(handlers, "inpaint_with_text"),
        ),
        Tool(
            name="create_reference",
            description=(
                "Generates a clean isolated reference image of one object using "
                "gpt-image-2 images.generate. Call once per object that needs "
                "identity preservation before inpaint_with_reference. Skip if a "
                "user-supplied reference image already exists. This does not advance "
                "the active canvas."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "object_name": {"type": "string"},
                    "size": {
                        "type": "string",
                        "description": "OpenAI image size string, usually 1024x1024.",
                    },
                },
                "required": ["description", "object_name"],
                "additionalProperties": False,
            },
            handler=_handler(handlers, "create_reference"),
        ),
        Tool(
            name="inpaint_with_reference",
            description=(
                "Reference-image-conditioned inpainting via gpt-image-2 images.edit. "
                "Use for identity or style consistency. The handler sends exactly "
                "three conceptual inputs: latest canvas, bbox mask, and reference "
                "image. The reference must exist first. Always pass the image_id "
                "returned by the immediately previous background, inpaint, or "
                "regenerate step. Render objects in the layout order only after "
                "validate_layout has approved that object's bbox; if the result says "
                "skipped with expected_object, handle that object next."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "image_id": {"type": "string"},
                    "bbox": _bbox_schema(),
                    "description": {"type": "string"},
                    "reference_image_id": {"type": "string"},
                    "object_name": {"type": "string"},
                },
                "required": [
                    "image_id",
                    "bbox",
                    "description",
                    "reference_image_id",
                    "object_name",
                ],
                "additionalProperties": False,
            },
            handler=_handler(handlers, "inpaint_with_reference"),
        ),
        Tool(
            name="validate_canvas",
            description=(
                "Post-inpaint vision check for the just-rendered object. Compare the "
                "current canvas against that single object's planned layout, including "
                "object position and bbox size/scale. Fix it with regenerate_object, "
                "using override_bbox when the object should be bigger, smaller, or moved, "
                "before moving to the next object. Do not validate all objects in parallel."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "image_id": {"type": "string"},
                    "object_name": {
                        "type": "string",
                        "description": (
                            "Optional just-rendered object name. Omit to validate the "
                            "current unaccepted object."
                        ),
                    },
                },
                "required": ["image_id"],
                "additionalProperties": False,
            },
            handler=_handler(handlers, "validate_canvas"),
        ),
        Tool(
            name="regenerate_object",
            description=(
                "Re-inpaints one object to fix a flagged issue. mode is 'text' for "
                "FLUX.1-Fill or 'reference' for gpt-image-2. Use optional overrides "
                "to change bbox position, bbox size, or prompt based on validator "
                "feedback. Max two retries per object. Always regenerate from the "
                "latest canvas image_id so previous accepted objects remain in the canvas."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "image_id": {"type": "string"},
                    "object_name": {"type": "string"},
                    "mode": {"type": "string", "enum": ["text", "reference"]},
                    "override_bbox": _bbox_schema(),
                    "override_description": {"type": "string"},
                },
                "required": ["image_id", "object_name", "mode"],
                "additionalProperties": False,
            },
            handler=_handler(handlers, "regenerate_object"),
        ),
        Tool(
            name="finish",
            description=(
                "Terminates the loop. Only call after validate_canvas returns ok: "
                "true or the bounded retry budget is exhausted. Finish with the "
                "latest canvas image_id only."
            ),
            input_schema={
                "type": "object",
                "properties": {"image_id": {"type": "string"}},
                "required": ["image_id"],
                "additionalProperties": False,
            },
            handler=_handler(handlers, "finish"),
            terminal=True,
        ),
    ]
