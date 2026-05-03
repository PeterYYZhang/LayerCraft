"""ChainArchitect agent."""

from __future__ import annotations

from typing import Any

from .llm import DEFAULT_LLM_MODEL, run_agent
from .tools import architect_tools


SYSTEM_PROMPT = (
    "You are ChainArchitect. Convert the prompt into a background and ordered foreground objects.\n"
    "Use only the provided tools and submit the layout when complete.\n"
    "The background description must be strictly environment-only: do not include "
    "foreground object words, synonyms, or related concepts. Put every requested "
    "foreground concept only in add_object calls.\n"
    "Choose a viewpoint that leaves clear, plausible space for all planned objects."
)


def _validate_bbox(bbox: list[int], canvas_size: tuple[int, int], name: str) -> None:
    if len(bbox) != 4:
        raise ValueError(f"Object '{name}' bbox must be [x, y, width, height]")
    x, y, width, height = [int(value) for value in bbox]
    if width <= 0 or height <= 0:
        raise ValueError(f"Object '{name}' bbox width and height must be positive")
    if x < 0 or y < 0 or x + width > canvas_size[0] or y + height > canvas_size[1]:
        raise ValueError(f"Object '{name}' bbox is outside the canvas")


def _validate_layout(layout: dict[str, Any], canvas_size: tuple[int, int]) -> dict[str, Any]:
    background = layout.get("background")
    objects = layout.get("objects")
    if not isinstance(background, dict):
        raise ValueError("Layout missing background")
    if not background.get("description") or not background.get("viewpoint"):
        raise ValueError("Background requires description and viewpoint")
    if not isinstance(objects, list):
        raise ValueError("Layout objects must be a list")

    normalized_objects = []
    for obj in objects:
        name = str(obj["name"])
        bbox = [int(value) for value in obj["bbox"]]
        _validate_bbox(bbox, canvas_size, name)
        normalized_objects.append(
            {
                "name": name,
                "description": str(obj["description"]),
                "bbox": bbox,
                "order": int(obj["order"]),
                "relations": list(obj.get("relations", [])),
                "requires_reference": bool(obj.get("requires_reference", False)),
            }
        )

    normalized_objects.sort(key=lambda item: item["order"])
    return {
        "canvas_size": [int(canvas_size[0]), int(canvas_size[1])],
        "background": {
            "description": str(background["description"]),
            "viewpoint": str(background["viewpoint"]),
        },
        "objects": normalized_objects,
    }


def plan_layout(
    enriched_prompt: str,
    canvas_size: tuple[int, int] = (1024, 1024),
    *,
    client: Any | None = None,
    model: str = DEFAULT_LLM_MODEL,
    recorder: Any | None = None,
) -> dict[str, Any]:
    """Run ChainArchitect and return a normalized layout dict."""

    state: dict[str, Any] = {"background": None, "objects": []}

    def set_background(args: dict[str, Any]) -> dict[str, Any]:
        state["background"] = {
            "description": args["description"],
            "viewpoint": args["viewpoint"],
        }
        return {"ok": True}

    def add_object(args: dict[str, Any]) -> dict[str, Any]:
        state["objects"].append(args)
        return {"ok": True, "count": len(state["objects"])}

    def submit_layout(_: dict[str, Any]) -> dict[str, Any]:
        layout = {
            "canvas_size": [canvas_size[0], canvas_size[1]],
            "background": state["background"],
            "objects": state["objects"],
        }
        return _validate_layout(layout, canvas_size)

    tools = architect_tools(
        {
            "set_background": set_background,
            "add_object": add_object,
            "submit_layout": submit_layout,
        }
    )
    user = (
        f"Prompt: {enriched_prompt}\n"
        f"Canvas size: {canvas_size[0]}x{canvas_size[1]} pixels.\n"
        "Plan all visible foreground objects with pixel bboxes.\n"
        "Set a background prompt that contains only the static environment and no "
        "words or related concepts from the foreground objects. Select a good camera "
        "viewpoint/framing that makes the object placements natural and visible."
    )
    result = run_agent(
        SYSTEM_PROMPT,
        user,
        tools,
        model=model,
        client=client,
        recorder=recorder,
        agent_name="ChainArchitect",
    )
    if not isinstance(result, dict):
        raise ValueError("ChainArchitect did not return a layout")
    return _validate_layout(result, canvas_size)
