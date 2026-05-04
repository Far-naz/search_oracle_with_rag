from src.helpers.text_normalization import normalize_name, strip_filler


def test_normalize_name_collapses_punctuation_and_case() -> None:
    assert normalize_name("  Alice-Smith, PhD!  ") == "alice smith phd"


def test_normalize_name_handles_empty_input() -> None:
    assert normalize_name("") == ""


def test_strip_filler_removes_advisor_working_on_wrapper() -> None:
    assert strip_filler("I am looking for advisor who is working on labor") == "labor"
