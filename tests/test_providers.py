"""Tests for response parsing in ``forminv.eval.providers``.

These are pure-function tests over raw model output strings; no network or API
keys are required.
"""

import pytest

from forminv.eval.providers import parse_label, parse_mc_label


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("TRUE", "TRUE"),
        ("FALSE", "FALSE"),
        ("true", "TRUE"),
        ("  False  ", "FALSE"),
        ("TRUE.", "TRUE"),
        ("FALSE -- because the converse fails", "FALSE"),
        ("The statement is TRUE", "TRUE"),
        ("Answer: TRUE", "TRUE"),
        ("This is false.", "FALSE"),
    ],
)
def test_parse_label_extracts_verdict(raw, expected):
    assert parse_label(raw) == expected


@pytest.mark.parametrize("raw", ["", "I am not sure", "maybe"])
def test_parse_label_invalid_returns_none(raw):
    assert parse_label(raw) is None


def test_parse_label_ambiguous_both_tokens_returns_none():
    # When both TRUE and FALSE appear with no leading verdict line, it is ambiguous.
    assert parse_label("It could be TRUE or FALSE") is None


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("A", "A"),
        ("B)", "B"),
        ("C.", "C"),
        ("The answer is D", "D"),
        ("d", "D"),
    ],
)
def test_parse_mc_label(raw, expected):
    assert parse_mc_label(raw) == expected


def test_parse_mc_label_invalid():
    assert parse_mc_label("none of these") is None
