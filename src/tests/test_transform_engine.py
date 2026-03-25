"""Tests for the transform engine."""

from src.execution.transform_engine import apply_transform
from src.models.query_plan import FilterCondition, Transform, TransformFilter


def _make_data():
    return {
        "users": [
            {"name": "alice", "age": 30, "active": True, "joined": "2024-01-15T00:00:00Z"},
            {"name": "bob", "age": 25, "active": False, "joined": "2024-06-20T00:00:00Z"},
            {"name": "carol", "age": 35, "active": True, "joined": "2023-03-10T00:00:00Z"},
        ]
    }


def test_filter_equals():
    """Filter with equals operator retains matching items."""
    transform = Transform(
        filter=TransformFilter(
            array_path="users",
            condition=FilterCondition(field="active", operator="equals", value=True),
        )
    )
    result = apply_transform(_make_data(), transform)
    assert len(result["users"]) == 2
    assert all(u["active"] for u in result["users"])


def test_filter_not_equals():
    """Filter with not_equals operator."""
    transform = Transform(
        filter=TransformFilter(
            array_path="users",
            condition=FilterCondition(field="name", operator="not_equals", value="bob"),
        )
    )
    result = apply_transform(_make_data(), transform)
    assert len(result["users"]) == 2
    assert "bob" not in [u["name"] for u in result["users"]]


def test_filter_greater_than():
    """Filter with greater_than operator."""
    transform = Transform(
        filter=TransformFilter(
            array_path="users",
            condition=FilterCondition(field="age", operator="greater_than", value=28),
        )
    )
    result = apply_transform(_make_data(), transform)
    assert len(result["users"]) == 2


def test_filter_less_than():
    """Filter with less_than operator."""
    transform = Transform(
        filter=TransformFilter(
            array_path="users",
            condition=FilterCondition(field="age", operator="less_than", value=30),
        )
    )
    result = apply_transform(_make_data(), transform)
    assert len(result["users"]) == 1
    assert result["users"][0]["name"] == "bob"


def test_filter_is_null():
    """Filter with is_null operator."""
    data = {"items": [{"val": None}, {"val": "something"}, {"val": None}]}
    transform = Transform(
        filter=TransformFilter(
            array_path="items",
            condition=FilterCondition(field="val", operator="is_null"),
        )
    )
    result = apply_transform(data, transform)
    assert len(result["items"]) == 2


def test_filter_is_not_null():
    """Filter with is_not_null operator."""
    data = {"items": [{"val": None}, {"val": "something"}, {"val": None}]}
    transform = Transform(
        filter=TransformFilter(
            array_path="items",
            condition=FilterCondition(field="val", operator="is_not_null"),
        )
    )
    result = apply_transform(data, transform)
    assert len(result["items"]) == 1


def test_filter_contains():
    """Filter with contains operator (string containment)."""
    transform = Transform(
        filter=TransformFilter(
            array_path="users",
            condition=FilterCondition(field="name", operator="contains", value="o"),
        )
    )
    result = apply_transform(_make_data(), transform)
    names = [u["name"] for u in result["users"]]
    assert "bob" in names
    assert "carol" in names
    assert "alice" not in names


def test_filter_before():
    """Filter with before operator (date string comparison)."""
    transform = Transform(
        filter=TransformFilter(
            array_path="users",
            condition=FilterCondition(
                field="joined", operator="before", value="2024-01-01T00:00:00Z"
            ),
        )
    )
    result = apply_transform(_make_data(), transform)
    assert len(result["users"]) == 1
    assert result["users"][0]["name"] == "carol"


def test_filter_after():
    """Filter with after operator (date string comparison)."""
    transform = Transform(
        filter=TransformFilter(
            array_path="users",
            condition=FilterCondition(
                field="joined", operator="after", value="2024-03-01T00:00:00Z"
            ),
        )
    )
    result = apply_transform(_make_data(), transform)
    assert len(result["users"]) == 1
    assert result["users"][0]["name"] == "bob"


def test_select_fields():
    """select_fields retains only specified fields."""
    transform = Transform(
        filter=TransformFilter(
            array_path="users",
            condition=FilterCondition(field="active", operator="equals", value=True),
        ),
        select_fields=["name", "age"],
    )
    result = apply_transform(_make_data(), transform)
    for user in result["users"]:
        assert set(user.keys()) == {"name", "age"}


def test_empty_result_after_filter():
    """Filter that matches nothing returns empty array."""
    transform = Transform(
        filter=TransformFilter(
            array_path="users",
            condition=FilterCondition(field="age", operator="greater_than", value=100),
        )
    )
    result = apply_transform(_make_data(), transform)
    assert result["users"] == []


def test_no_mutation_of_original():
    """Transform does not mutate the original data."""
    data = _make_data()
    original_count = len(data["users"])
    transform = Transform(
        filter=TransformFilter(
            array_path="users",
            condition=FilterCondition(field="active", operator="equals", value=True),
        )
    )
    apply_transform(data, transform)
    assert len(data["users"]) == original_count
