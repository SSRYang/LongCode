from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class ToolResult:
    content: str
    is_error: bool = False


class ToolInputValidationError(ValueError):
    pass


class Tool(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    @property
    @abstractmethod
    def input_schema(self) -> dict: ...

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult: ...

    def get_activity_description(self, **kwargs) -> str | None:
        return None

    def is_read_only(self) -> bool:
        return False

    def to_api_schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }

    def validate_input(self, tool_input: Any) -> str | None:
        try:
            _validate_schema_value(tool_input, self.input_schema)
        except ToolInputValidationError as exc:
            return str(exc)
        return None


def _validate_schema_value(value: Any, schema: dict[str, Any], path: str = "") -> None:
    if not isinstance(schema, dict):
        return

    schema_type = schema.get("type")
    if schema_type == "object":
        _validate_object(value, schema, path)
    elif schema_type == "array":
        _validate_array(value, schema, path)
    elif schema_type == "string":
        if not isinstance(value, str):
            raise ToolInputValidationError(
                f"expected string for {_path_label(path)}, got {_value_type(value)}"
            )
    elif schema_type == "integer":
        if not isinstance(value, int) or isinstance(value, bool):
            raise ToolInputValidationError(
                f"expected integer for {_path_label(path)}, got {_value_type(value)}"
            )
    elif schema_type == "boolean":
        if not isinstance(value, bool):
            raise ToolInputValidationError(
                f"expected boolean for {_path_label(path)}, got {_value_type(value)}"
            )

    enum_values = schema.get("enum")
    if enum_values is not None and value not in enum_values:
        raise ToolInputValidationError(
            f"invalid value for {_path_label(path)}: expected one of {list(enum_values)}"
        )


def _validate_object(value: Any, schema: dict[str, Any], path: str) -> None:
    if not isinstance(value, dict):
        raise ToolInputValidationError(
            f"expected object for {_path_label(path)}, got {_value_type(value)}"
        )

    properties = schema.get("properties")
    properties = properties if isinstance(properties, dict) else {}

    required = schema.get("required")
    required = required if isinstance(required, list) else []
    for field in required:
        if field not in value:
            raise ToolInputValidationError(
                f"missing required field {_path_label(_child_path(path, str(field)))}"
            )

    for key in value:
        if key not in properties:
            raise ToolInputValidationError(
                f"unexpected field {_path_label(_child_path(path, str(key)))}"
            )

    for key, child_schema in properties.items():
        if key in value:
            _validate_schema_value(value[key], child_schema, _child_path(path, str(key)))


def _validate_array(value: Any, schema: dict[str, Any], path: str) -> None:
    if not isinstance(value, list):
        raise ToolInputValidationError(
            f"expected array for {_path_label(path)}, got {_value_type(value)}"
        )

    min_items = schema.get("minItems")
    if isinstance(min_items, int) and len(value) < min_items:
        raise ToolInputValidationError(
            f"expected at least {min_items} items for {_path_label(path)}, got {len(value)}"
        )

    max_items = schema.get("maxItems")
    if isinstance(max_items, int) and len(value) > max_items:
        raise ToolInputValidationError(
            f"expected at most {max_items} items for {_path_label(path)}, got {len(value)}"
        )

    item_schema = schema.get("items")
    if isinstance(item_schema, dict):
        for index, item in enumerate(value):
            _validate_schema_value(item, item_schema, _index_path(path, index))


def _child_path(path: str, child: str) -> str:
    return child if not path else f"{path}.{child}"


def _index_path(path: str, index: int) -> str:
    return f"[{index}]" if not path else f"{path}[{index}]"


def _path_label(path: str) -> str:
    return f"'{path}'" if path else "input"


def _value_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__
