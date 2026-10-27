from __future__ import annotations

from typing import Literal

import pytest

from agentloop import ToolRegistry, tool
from agentloop.errors import SchemaError, ToolError
from agentloop.messages import ToolCall
from agentloop.tools import build_schema


def test_schema_primitives_and_required():
    @tool
    def f(name: str, age: int, score: float, active: bool) -> str:
        "Describe a person."
        return name

    schema = f.tool_spec.input_schema
    assert schema["type"] == "object"
    assert schema["properties"]["name"] == {"type": "string"}
    assert schema["properties"]["age"] == {"type": "integer"}
    assert schema["properties"]["score"] == {"type": "number"}
    assert schema["properties"]["active"] == {"type": "boolean"}
    assert set(schema["required"]) == {"name", "age", "score", "active"}
    assert f.tool_spec.description == "Describe a person."


def test_defaults_are_optional():
    @tool
    def f(a: int, b: int = 2) -> int:
        "Add."
        return a + b

    schema = f.tool_spec.input_schema
    assert schema["required"] == ["a"]
    assert "b" in schema["properties"]


def test_optional_annotation_not_required():
    @tool
    def f(a: int, b: str | None = None) -> str:
        "Thing."
        return str(a)

    schema = f.tool_spec.input_schema
    assert schema["required"] == ["a"]
    # Optional[str] unwraps to a plain string schema.
    assert schema["properties"]["b"] == {"type": "string"}


def test_literal_becomes_enum():
    @tool
    def f(unit: Literal["c", "f"] = "c") -> str:
        "Unit."
        return unit

    prop = f.tool_spec.input_schema["properties"]["unit"]
    assert prop["enum"] == ["c", "f"]
    assert prop["type"] == "string"


def test_list_items_schema():
    @tool
    def f(xs: list[int]) -> int:
        "Sum."
        return sum(xs)

    prop = f.tool_spec.input_schema["properties"]["xs"]
    assert prop == {"type": "array", "items": {"type": "integer"}}


def test_docstring_arg_descriptions():
    @tool
    def get_weather(city: str, unit: str = "c") -> str:
        """Look up the weather.

        Args:
            city: Name of the city.
            unit: Temperature unit (c or f).
        """
        return city

    props = get_weather.tool_spec.input_schema["properties"]
    assert get_weather.tool_spec.description == "Look up the weather."
    assert props["city"]["description"] == "Name of the city."
    assert props["unit"]["description"] == "Temperature unit (c or f)."


def test_custom_name_and_description():
    @tool(name="renamed", description="Custom.")
    def f(a: int) -> int:
        "Original."
        return a

    assert f.tool_spec.name == "renamed"
    assert f.tool_spec.description == "Custom."


def test_varargs_rejected():
    with pytest.raises(SchemaError):
        @tool
        def f(*args: int) -> int:
            "bad"
            return 0


def test_registry_execute_success():
    @tool
    def add(a: int, b: int) -> int:
        "Add."
        return a + b

    reg = ToolRegistry([add])
    result = reg.execute(ToolCall(id="1", name="add", arguments={"a": 2, "b": 3}))
    assert result.content == "5"
    assert result.is_error is False


def test_registry_unknown_tool_is_error_not_raise():
    reg = ToolRegistry([])
    result = reg.execute(ToolCall(id="1", name="nope", arguments={}))
    assert result.is_error is True
    assert "unknown tool" in result.content


def test_registry_tool_exception_is_error():
    @tool
    def boom() -> str:
        "Explode."
        raise ValueError("kaboom")

    reg = ToolRegistry([boom])
    result = reg.execute(ToolCall(id="1", name="boom", arguments={}))
    assert result.is_error is True
    assert "kaboom" in result.content


def test_registry_bad_arguments_is_error():
    @tool
    def add(a: int, b: int) -> int:
        "Add."
        return a + b

    reg = ToolRegistry([add])
    result = reg.execute(ToolCall(id="1", name="add", arguments={"a": 1}))
    assert result.is_error is True


def test_duplicate_registration_raises():
    @tool
    def a() -> int:
        "a"
        return 1

    reg = ToolRegistry([a])
    with pytest.raises(ToolError):
        reg.register(a)


def test_non_str_result_is_json_encoded():
    @tool
    def payload() -> dict:
        "Return a dict."
        return {"ok": True, "n": 3}

    reg = ToolRegistry([payload])
    result = reg.execute(ToolCall(id="1", name="payload", arguments={}))
    assert result.content == '{"ok": true, "n": 3}'


def test_build_schema_standalone():
    def f(a: int) -> int:
        "Standalone."
        return a

    spec = build_schema(f)
    assert spec.name == "f"
    assert spec.input_schema["required"] == ["a"]
