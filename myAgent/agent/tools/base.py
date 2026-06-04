
from abc import ABC, abstractmethod
from collections.abc import Callable
from copy import deepcopy
from typing import Any, TypeVar

_ToolT = TypeVar("_ToolT", bound="Tool")


_JSON_TYPE_MAP: dict[str, type | tuple[type, ...]] = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
}

class Schema(ABC):
    """Abstract base for JSON Schema fragments describing tool parameters.

    Concrete types live in :mod:`nanobot.agent.tools.schema`; all implement
    :meth:`to_json_schema` and :meth:`validate_value`. Class methods
    :meth:`validate_json_schema_value` and :meth:`fragment` are the shared validation and normalization entry points.
    """
    @staticmethod
    def resolve_json_schema_type(t: Any) -> str | None:
        """Resolve the non-null type name from JSON Schema ``type`` (e.g. ``['string','null']`` -> ``'string'``)."""
        if isinstance(t, list):
            return next((x for x in t if x != "null"), None)
        return t  # type: ignore[return-value]

    @staticmethod
    def subpath(path: str, key: str) -> str:
        return f"{path}.{key}" if path else key

    @staticmethod
    def validate_json_schema_value(val: Any, schema: dict[str, Any], path: str = "") -> list[str]:
        """Validate ``val`` against a JSON Schema fragment; returns error messages (empty means valid).

        Used by :class:`Tool` and each concrete Schema's :meth:`validate_value`.
        """
        raw_type = schema.get("type")
        nullable = (isinstance(raw_type, list) and "null" in raw_type) or schema.get("nullable", False)
        t = Schema.resolve_json_schema_type(raw_type)
        label = path or "parameter"

        if nullable and val is None:
            return []
        if t == "integer" and (not isinstance(val, int) or isinstance(val, bool)):
            return [f"{label} should be integer"]
        if t == "number" and (
            not isinstance(val, _JSON_TYPE_MAP["number"]) or isinstance(val, bool)
        ):
            return [f"{label} should be number"]
        if t in _JSON_TYPE_MAP and t not in ("integer", "number") and not isinstance(val, _JSON_TYPE_MAP[t]):
            return [f"{label} should be {t}"]

        errors: list[str] = []
        if "enum" in schema and val not in schema["enum"]:
            errors.append(f"{label} must be one of {schema['enum']}")
        if t in ("integer", "number"):
            if "minimum" in schema and val < schema["minimum"]:
                errors.append(f"{label} must be >= {schema['minimum']}")
            if "maximum" in schema and val > schema["maximum"]:
                errors.append(f"{label} must be <= {schema['maximum']}")
        if t == "string":
            if "minLength" in schema and len(val) < schema["minLength"]:
                errors.append(f"{label} must be at least {schema['minLength']} chars")
            if "maxLength" in schema and len(val) > schema["maxLength"]:
                errors.append(f"{label} must be at most {schema['maxLength']} chars")
        if t == "object":
            props = schema.get("properties", {})
            for k in schema.get("required", []):
                if k not in val:
                    errors.append(f"missing required {Schema.subpath(path, k)}")
            for k, v in val.items():
                if k in props:
                    errors.extend(Schema.validate_json_schema_value(v, props[k], Schema.subpath(path, k)))
        if t == "array":
            if "minItems" in schema and len(val) < schema["minItems"]:
                errors.append(f"{label} must have at least {schema['minItems']} items")
            if "maxItems" in schema and len(val) > schema["maxItems"]:
                errors.append(f"{label} must be at most {schema['maxItems']} items")
            if "items" in schema:
                prefix = f"{path}[{{}}]" if path else "[{}]"
                for i, item in enumerate(val):
                    errors.extend(
                        Schema.validate_json_schema_value(item, schema["items"], prefix.format(i))
                    )
        return errors

    @abstractmethod
    def to_json_schema(self) -> dict[str, Any]:
        """Return a fragment dict compatible with :meth:`validate_json_schema_value`."""
        ...

    def validate_value(self, value: Any, path: str = "") -> list[str]:
        """Validate a single value; returns error messages (empty means pass). Subclasses may override for extra rules."""
        return Schema.validate_json_schema_value(value, self.to_json_schema(), path)

class Tool(ABC):
    """Agent capability: read files, run commands, etc."""
    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name used in function calls."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Description of what the tool does."""
        ...

    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        """JSON Schema for tool parameters."""
        ...

    @abstractmethod
    async def execute(self, **kwargs: Any) -> Any:
        """Run the tool; returns a string or list of content blocks."""
        ...

def tool_parameters(schema: dict[str, Any]) -> Callable[[type[_ToolT]], type[_ToolT]]:
    """Class decorator: attach JSON Schema and inject a concrete ``parameters`` property.

    Use on ``Tool`` subclasses instead of writing ``@property def parameters``. The
    schema is stored on the class and returned as a fresh copy on each access.

    Example::

        @tool_parameters({
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        })
        class ReadFileTool(Tool):
            ...
    """

    def decorator(cls: type[_ToolT]) -> type[_ToolT]:
        frozen = deepcopy(schema)

        @property
        def parameters(self: Any) -> dict[str, Any]:
            return deepcopy(frozen)

        cls.parameters = parameters  # type: ignore[assignment]

        abstract = getattr(cls, "__abstractmethods__", None)
        if abstract is not None and "parameters" in abstract:
            cls.__abstractmethods__ = frozenset(abstract - {"parameters"})  # type: ignore[misc]

        return cls

    return decorator
