from src.helpers.json_utils import safe_json_loads


def test_safe_json_loads_returns_empty_for_none_or_empty() -> None:
    assert safe_json_loads(None) == []
    assert safe_json_loads("") == []


def test_safe_json_loads_returns_empty_for_invalid_json() -> None:
    assert safe_json_loads("not-json") == []


def test_safe_json_loads_returns_empty_for_non_list_json() -> None:
    assert safe_json_loads('{"a": 1}') == []


def test_safe_json_loads_returns_list_for_valid_list_json() -> None:
    assert safe_json_loads('["x", "y"]') == ["x", "y"]
