"""Tests for the template resolver."""

import pytest

from src.execution.template_resolver import TemplateResolutionError, resolve_parameters


def test_resolve_simple_reference():
    """Resolves {{step_1.data}} from step results."""
    result = resolve_parameters(
        {"input": "{{step_1.users}}"},
        {"step_1": {"users": [{"name": "alice"}]}},
    )
    assert result["input"] == [{"name": "alice"}]


def test_resolve_nested_path():
    """Resolves {{step_1.data.nested.value}} via dot-path navigation."""
    result = resolve_parameters(
        {"value": "{{step_1.data.nested.count}}"},
        {"step_1": {"data": {"nested": {"count": 42}}}},
    )
    assert result["value"] == 42


def test_resolve_list_index():
    """Resolves {{step_1.users.0.name}} with list indexing."""
    result = resolve_parameters(
        {"first_user": "{{step_1.users.0.name}}"},
        {"step_1": {"users": [{"name": "alice"}, {"name": "bob"}]}},
    )
    assert result["first_user"] == "alice"


def test_resolve_item_reference():
    """Resolves {{repo.name}} in iterate_over context."""
    result = resolve_parameters(
        {"repo_name": "{{repo.name}}"},
        {},
        iteration_item={"name": "my-repo", "id": 123},
        item_alias="repo",
    )
    assert result["repo_name"] == "my-repo"


def test_resolve_multiple_templates_in_string():
    """Resolves multiple templates within a single string value."""
    result = resolve_parameters(
        {"message": "User {{step_1.name}} has {{step_1.count}} items"},
        {"step_1": {"name": "alice", "count": 5}},
    )
    assert result["message"] == "User alice has 5 items"


def test_resolve_missing_step_raises():
    """Missing step_id in references raises TemplateResolutionError."""
    with pytest.raises(TemplateResolutionError, match="step_99"):
        resolve_parameters(
            {"input": "{{step_99.data}}"},
            {"step_1": {"data": "value"}},
        )


def test_resolve_missing_path_raises():
    """Missing path in step data raises TemplateResolutionError."""
    with pytest.raises(TemplateResolutionError, match="not found"):
        resolve_parameters(
            {"input": "{{step_1.nonexistent}}"},
            {"step_1": {"data": "value"}},
        )


def test_resolve_non_string_passthrough():
    """Non-string values pass through unchanged."""
    result = resolve_parameters(
        {"count": 42, "flag": True, "items": [1, 2, 3]},
        {},
    )
    assert result == {"count": 42, "flag": True, "items": [1, 2, 3]}


def test_resolve_nested_dict_parameters():
    """Templates in nested dict values are resolved."""
    result = resolve_parameters(
        {"outer": {"inner": "{{step_1.value}}"}},
        {"step_1": {"value": "resolved"}},
    )
    assert result["outer"]["inner"] == "resolved"


def test_resolve_no_templates():
    """Parameters without templates are returned as-is."""
    params = {"key": "plain_value", "num": 10}
    result = resolve_parameters(params, {})
    assert result == params
