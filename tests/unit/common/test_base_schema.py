"""Tests for BaseSchema serialization and ID generation."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import Enum

import pytest

from src.common.base_schema import (
    BaseSchema,
    _canon,
    _qfloat,
    deterministic_id_from_dataclass,
)


class SampleEnum(Enum):
    """Sample enum for serialization tests."""
    VALUE_A = "a"
    VALUE_B = "b"


@dataclass
class SimpleSchema(BaseSchema):
    """Simple schema for testing."""
    name: str
    value: int


@dataclass
class NestedSchema(BaseSchema):
    """Schema with nested dataclass."""
    label: str
    inner: SimpleSchema


@dataclass
class SchemaWithEnum(BaseSchema):
    """Schema with enum field."""
    name: str
    kind: SampleEnum


@dataclass
class SchemaWithPrivate(BaseSchema):
    """Schema with private fields."""
    public_value: int
    _private_value: int = 0


@dataclass
class SchemaWithSpecialFloats(BaseSchema):
    """Schema with special float values."""
    normal: float
    nan_value: float
    inf_value: float
    neg_inf_value: float


@dataclass
class SchemaWithOptional(BaseSchema):
    """Schema with optional fields."""
    required: str
    optional: str | None = None
    optional_int: int | None = None


@dataclass
class SchemaWithLists(BaseSchema):
    """Schema with list fields."""
    items: list[int]
    nested_items: list[SimpleSchema] = field(default_factory=list)


class TestQfloat:
    """Tests for _qfloat stable rounding."""

    def test_normal_float(self):
        result = _qfloat(3.14159265358979, places=4)
        assert result == 3.1416

    def test_nan_returns_zero(self):
        result = _qfloat(float("nan"))
        assert result == 0.0

    def test_positive_inf_returns_large_finite(self):
        result = _qfloat(float("inf"))
        assert result == 1e10

    def test_negative_inf_returns_large_negative(self):
        result = _qfloat(float("-inf"))
        assert result == -1e10

    def test_negative_zero_normalized(self):
        result = _qfloat(-0.0)
        assert result == 0.0
        # Ensure it's not -0.0
        assert str(result) == "0.0"


class TestCanon:
    """Tests for _canon canonicalization."""

    def test_float_canonicalization(self):
        result = _canon(3.14159265)
        assert isinstance(result, float)
        assert abs(result - 3.14159265) < 1e-7

    def test_nan_to_string(self):
        result = _canon(float("nan"))
        assert result == "NaN"

    def test_inf_to_string(self):
        result = _canon(float("inf"))
        assert result == "Inf"

    def test_neg_inf_to_string(self):
        result = _canon(float("-inf"))
        assert result == "-Inf"

    def test_enum_to_value(self):
        result = _canon(SampleEnum.VALUE_A)
        assert result == "a"

    def test_dataclass_canonicalization(self):
        obj = SimpleSchema(name="test", value=42)
        result = _canon(obj)
        assert result == {"name": "test", "value": 42}

    def test_private_fields_excluded(self):
        obj = SchemaWithPrivate(public_value=1, _private_value=99)
        result = _canon(obj)
        assert "public_value" in result
        assert "_private_value" not in result

    def test_list_canonicalization(self):
        result = _canon([1.5, 2.5, 3.5])
        assert len(result) == 3

    def test_max_list_length_truncation(self):
        result = _canon(list(range(100)), max_list_length=5)
        assert result == "[100 items]"

    def test_max_string_length_truncation(self):
        long_string = "a" * 1000
        result = _canon(long_string, max_string_length=50)
        assert result.endswith("...[1000 chars]")
        assert len(result) < len(long_string)


class TestDeterministicId:
    """Tests for deterministic_id_from_dataclass."""

    def test_same_values_same_id(self):
        obj1 = SimpleSchema(name="test", value=42)
        obj2 = SimpleSchema(name="test", value=42)
        assert deterministic_id_from_dataclass(obj1) == deterministic_id_from_dataclass(obj2)

    def test_different_values_different_id(self):
        obj1 = SimpleSchema(name="test", value=42)
        obj2 = SimpleSchema(name="test", value=43)
        assert deterministic_id_from_dataclass(obj1) != deterministic_id_from_dataclass(obj2)

    def test_id_is_hex_string(self):
        obj = SimpleSchema(name="test", value=42)
        id_str = deterministic_id_from_dataclass(obj)
        assert isinstance(id_str, str)
        # Should be valid hex
        int(id_str, 16)

    def test_nested_dataclass_id(self):
        inner = SimpleSchema(name="inner", value=1)
        obj = NestedSchema(label="outer", inner=inner)
        id_str = deterministic_id_from_dataclass(obj)
        assert isinstance(id_str, str)


class TestBaseSchemaGetId:
    """Tests for BaseSchema.get_id()."""

    def test_get_id_returns_string(self):
        obj = SimpleSchema(name="test", value=42)
        assert isinstance(obj.get_id(), str)

    def test_get_id_deterministic(self):
        obj = SimpleSchema(name="test", value=42)
        assert obj.get_id() == obj.get_id()


class TestBaseSchemaToDict:
    """Tests for BaseSchema.to_dict()."""

    def test_simple_to_dict(self):
        obj = SimpleSchema(name="test", value=42)
        result = obj.to_dict()
        assert result == {"name": "test", "value": 42}

    def test_nested_to_dict(self):
        inner = SimpleSchema(name="inner", value=1)
        obj = NestedSchema(label="outer", inner=inner)
        result = obj.to_dict()
        assert result == {"label": "outer", "inner": {"name": "inner", "value": 1}}

    def test_enum_to_dict(self):
        obj = SchemaWithEnum(name="test", kind=SampleEnum.VALUE_A)
        result = obj.to_dict()
        assert result == {"name": "test", "kind": "a"}

    def test_private_fields_excluded(self):
        obj = SchemaWithPrivate(public_value=1, _private_value=99)
        result = obj.to_dict()
        assert "public_value" in result
        assert "_private_value" not in result

    def test_special_floats_handled(self):
        obj = SchemaWithSpecialFloats(
            normal=1.5,
            nan_value=float("nan"),
            inf_value=float("inf"),
            neg_inf_value=float("-inf"),
        )
        result = obj.to_dict()
        assert result["normal"] == 1.5
        assert result["nan_value"] == "NaN"
        assert result["inf_value"] == "Inf"
        assert result["neg_inf_value"] == "-Inf"

    def test_max_list_length(self):
        obj = SchemaWithLists(items=list(range(100)))
        result = obj.to_dict(max_list_length=5)
        assert result["items"] == "[100 items]"

    def test_max_string_length(self):
        long_name = "x" * 1000
        obj = SimpleSchema(name=long_name, value=1)
        result = obj.to_dict(max_string_length=50)
        assert result["name"].endswith("...[1000 chars]")


class TestBaseSchemaFromDict:
    """Tests for BaseSchema.from_dict()."""

    def test_simple_from_dict(self):
        data = {"name": "test", "value": 42}
        obj = SimpleSchema.from_dict(data)
        assert obj.name == "test"
        assert obj.value == 42

    def test_nested_from_dict(self):
        data = {"label": "outer", "inner": {"name": "inner", "value": 1}}
        obj = NestedSchema.from_dict(data)
        assert obj.label == "outer"
        assert isinstance(obj.inner, SimpleSchema)
        assert obj.inner.name == "inner"
        assert obj.inner.value == 1

    def test_enum_from_dict(self):
        data = {"name": "test", "kind": "a"}
        obj = SchemaWithEnum.from_dict(data)
        assert obj.kind == SampleEnum.VALUE_A

    def test_optional_fields_from_dict(self):
        data = {"required": "test"}
        obj = SchemaWithOptional.from_dict(data)
        assert obj.required == "test"
        assert obj.optional is None

    def test_optional_fields_with_values(self):
        data = {"required": "test", "optional": "value", "optional_int": 42}
        obj = SchemaWithOptional.from_dict(data)
        assert obj.optional == "value"
        assert obj.optional_int == 42

    def test_list_of_dataclasses_from_dict(self):
        data = {
            "items": [1, 2, 3],
            "nested_items": [{"name": "a", "value": 1}, {"name": "b", "value": 2}],
        }
        obj = SchemaWithLists.from_dict(data)
        assert obj.items == [1, 2, 3]
        assert len(obj.nested_items) == 2
        assert all(isinstance(item, SimpleSchema) for item in obj.nested_items)


class TestBaseSchemaToString:
    """Tests for BaseSchema.to_string()."""

    def test_to_string_is_json(self):
        obj = SimpleSchema(name="test", value=42)
        result = obj.to_string()
        # Should be valid JSON
        parsed = json.loads(result)
        assert parsed["name"] == "test"

    def test_to_string_truncates_long_lists(self):
        obj = SchemaWithLists(items=list(range(100)))
        result = obj.to_string(max_list_length=5)
        assert "[100 items]" in result


class TestBaseSchemaStr:
    """Tests for BaseSchema.__str__()."""

    def test_str_returns_string(self):
        obj = SimpleSchema(name="test", value=42)
        result = str(obj)
        assert isinstance(result, str)
        # Should be JSON
        json.loads(result)
