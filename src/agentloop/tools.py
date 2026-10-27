"""Turn ordinary typed Python functions into LLM tools.

A function decorated with :func:`tool` carries a JSON schema derived from its
signature and docstring, so the model can call it with validated arguments and
the agent can dispatch the call. No manual schema writing, no base classes.

    @tool
    def get_weather(city: str, unit: Literal["c", "f"] = "c") -> str:
        '''Look up the current weather.

        Args:
            city: Name of the city.
            unit: Temperature unit.
        '''
        ...
"""
from __future__ import annotations

import inspect
import json
import re
import typing
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, get_args, get_origin

from .errors import SchemaError, ToolError
from .messages import ToolCall, ToolResult

_PRIMITIVES: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}

# Docstring section headers that mark the end of the Args: block.
_DOC_SECTION_ENDS = {
    "returns", "return", "raises", "yields", "examples", "example", "notes", "note",
}


def _is_union(origin: Any) -> bool:
    """True for both ``typing.Union[...]`` and PEP 604 ``X | Y`` origins."""
    return origin is typing.Union or getattr(origin, "__name__", "") == "UnionType"


def _strip_optional(annotation: Any) -> tuple[Any, bool]:
    """Unwrap ``Optional[X]`` / ``X | None`` into ``(X, is_optional)``."""
    if _is_union(get_origin(annotation)):
        args = get_args(annotation)
        non_none = [a for a in args if a is not type(None)]
        is_optional = len(non_none) != len(args)
        if len(non_none) == 1:
            return non_none[0], is_optional
        # A genuine multi-type union — handed back as-is (schema falls through to {}).
        return annotation, is_optional
    return annotation, False


def _schema_for(annotation: Any) -> dict:
    """Map a Python type annotation to a JSON-schema fragment."""
    annotation, _ = _strip_optional(annotation)

    if annotation is inspect.Parameter.empty or annotation is Any:
        return {}  # unconstrained

    if annotation in _PRIMITIVES:
        return {"type": _PRIMITIVES[annotation]}

    origin = get_origin(annotation)

    if origin is typing.Literal:
        choices = list(get_args(annotation))
        schema: dict = {"enum": choices}
        if choices and type(choices[0]) in _PRIMITIVES:
            schema["type"] = _PRIMITIVES[type(choices[0])]
        return schema

    if origin in (list, set, tuple, frozenset):
        args = get_args(annotation)
        items = _schema_for(args[0]) if args else {}
        return {"type": "array", "items": items}

    if origin is dict:
        return {"type": "object"}

    if annotation in (list, set, tuple):
        return {"type": "array"}
    if annotation is dict:
        return {"type": "object"}

    # Unknown / multi-type union: accept anything rather than reject the tool.
    return {}


def _parse_docstring(doc: str | None) -> tuple[str, dict[str, str]]:
    """Return ``(summary, {arg: description})`` from a Google-style docstring."""
    if not doc:
        return "", {}
    lines = inspect.cleandoc(doc).splitlines()

    summary: list[str] = []
    arg_docs: dict[str, str] = {}
    in_args = False
    summary_done = False
    current: str | None = None

    for line in lines:
        header = line.strip().rstrip(":").lower()
        if header in {"args", "arguments", "parameters", "params"}:
            in_args = True
            continue
        if in_args and header in _DOC_SECTION_ENDS:
            break
        if not in_args:
            stripped = line.strip()
            if not stripped:
                if summary:
                    summary_done = True  # summary ends at the first blank line...
                continue                 # ...but keep scanning for an Args: section
            if not summary_done:
                summary.append(stripped)
            continue
        m = re.match(r"\s*([*\w]+)\s*(?:\([^)]*\))?\s*:\s*(.*)", line)
        if m:
            current = m.group(1).lstrip("*")
            arg_docs[current] = m.group(2).strip()
        elif current and line.strip():
            arg_docs[current] += " " + line.strip()

    return " ".join(summary), arg_docs


@dataclass
class ToolSpec:
    """A callable plus the JSON schema that describes how to call it."""

    name: str
    description: str
    input_schema: dict
    func: Callable[..., Any]

    def to_schema(self) -> dict:
        """The vendor-neutral tool definition (name/description/input_schema)."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


def build_schema(func: Callable[..., Any]) -> ToolSpec:
    """Introspect ``func`` and build its :class:`ToolSpec`."""
    sig = inspect.signature(func)
    try:
        hints = typing.get_type_hints(func)
    except Exception as exc:  # pragma: no cover - exotic annotations
        raise SchemaError(f"Cannot resolve type hints for {func.__name__!r}: {exc}") from exc

    summary, arg_docs = _parse_docstring(func.__doc__)

    properties: dict[str, dict] = {}
    required: list[str] = []

    for pname, param in sig.parameters.items():
        if pname in {"self", "cls"}:
            continue
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            raise SchemaError(f"{func.__name__!r}: *args/**kwargs tools are not supported")

        annotation = hints.get(pname, param.annotation)
        prop = _schema_for(annotation)
        if pname in arg_docs:
            prop = {**prop, "description": arg_docs[pname]}
        properties[pname] = prop

        if param.default is inspect.Parameter.empty:
            required.append(pname)

    input_schema: dict = {"type": "object", "properties": properties}
    if required:
        input_schema["required"] = required

    return ToolSpec(
        name=func.__name__,
        description=summary or (func.__doc__ or "").strip(),
        input_schema=input_schema,
        func=func,
    )


def tool(
    func: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Any:
    """Decorator that attaches a :class:`ToolSpec` to a function.

    Usable bare (``@tool``) or parameterized (``@tool(name=..., description=...)``).
    The wrapped function stays callable as normal; its spec lives on
    ``func.tool_spec``.
    """

    def decorate(fn: Callable[..., Any]) -> Callable[..., Any]:
        spec = build_schema(fn)
        if name:
            spec.name = name
        if description:
            spec.description = description
        fn.tool_spec = spec  # type: ignore[attr-defined]
        return fn

    return decorate(func) if func is not None else decorate


def _coerce_to_str(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, default=str)
    except (TypeError, ValueError):
        return str(value)


class ToolRegistry:
    """Holds the tools available to an agent and dispatches calls to them."""

    def __init__(self, tools: list[Callable[..., Any]] | None = None) -> None:
        self._specs: dict[str, ToolSpec] = {}
        for t in tools or []:
            self.register(t)

    def register(self, fn: Callable[..., Any]) -> ToolSpec:
        spec: ToolSpec = getattr(fn, "tool_spec", None) or build_schema(fn)
        if spec.name in self._specs:
            raise ToolError(f"Duplicate tool name: {spec.name!r}")
        self._specs[spec.name] = spec
        return spec

    def __contains__(self, name: object) -> bool:
        return name in self._specs

    def __len__(self) -> int:
        return len(self._specs)

    @property
    def names(self) -> list[str]:
        return list(self._specs)

    def specs(self) -> list[ToolSpec]:
        return list(self._specs.values())

    def schemas(self) -> list[dict]:
        return [s.to_schema() for s in self._specs.values()]

    def execute(self, call: ToolCall) -> ToolResult:
        """Run one tool call, capturing failures as an error result.

        Tool errors are returned to the model (not raised) so the agent can see
        what went wrong and recover on the next step.
        """
        spec = self._specs.get(call.name)
        if spec is None:
            return ToolResult(
                tool_call_id=call.id,
                content=f"Error: unknown tool {call.name!r}. Available: {self.names}",
                is_error=True,
            )
        try:
            result = spec.func(**call.arguments)
        except TypeError as exc:
            msg = f"Error: bad arguments for {call.name!r}: {exc}"
            return ToolResult(call.id, msg, is_error=True)
        except Exception as exc:  # noqa: BLE001 - surfaced to the model, not swallowed
            return ToolResult(call.id, f"Error from {call.name!r}: {exc!r}", is_error=True)
        return ToolResult(call.id, _coerce_to_str(result), is_error=False)
