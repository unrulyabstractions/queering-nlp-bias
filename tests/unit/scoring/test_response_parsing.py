"""Tests for response parsing utilities in scoring."""

from __future__ import annotations

import re

import pytest


# These are example parsing functions that match common patterns
# in LLM response parsing for scoring


def parse_categorical_response(text: str, categories: list[str]) -> str | None:
    """Parse a categorical response from text.

    Looks for category names in the response.
    Returns the first matching category, or None if no match.
    """
    text_lower = text.lower().strip()

    # Direct yes/no mapping for binary categories
    if set(categories) == {"yes", "no"} or set(categories) == {0, 1}:
        # Check for common yes indicators
        if text_lower in ("yes", "1", "true", "correct"):
            return "yes" if "yes" in categories else 1
        if text_lower in ("no", "0", "false", "incorrect"):
            return "no" if "no" in categories else 0

    # Look for category in response
    for cat in categories:
        if str(cat).lower() in text_lower:
            return cat

    return None


def parse_graded_response(text: str, scale_min: int, scale_max: int) -> int | None:
    """Parse a graded (numeric) response from text.

    Extracts the first number in the valid range.
    """
    # Find all numbers in the text
    numbers = re.findall(r'\d+', text)

    for num_str in numbers:
        num = int(num_str)
        if scale_min <= num <= scale_max:
            return num

    return None


def strip_thinking_block(text: str) -> str:
    """Remove <think>...</think> blocks from response."""
    pattern = r"<think>.*?</think>\s*"
    return re.sub(pattern, "", text, flags=re.DOTALL)


class TestCategoricalParsing:
    """Tests for categorical response parsing."""

    def test_parse_categorical_0(self):
        result = parse_categorical_response("0", [0, 1])
        assert result == 0

    def test_parse_categorical_1(self):
        result = parse_categorical_response("1", [0, 1])
        assert result == 1

    def test_parse_categorical_yes(self):
        result = parse_categorical_response("yes", ["yes", "no"])
        assert result == "yes"

    def test_parse_categorical_no(self):
        result = parse_categorical_response("no", ["yes", "no"])
        assert result == "no"

    def test_parse_categorical_case_insensitive(self):
        result = parse_categorical_response("YES", ["yes", "no"])
        assert result == "yes"

    def test_parse_categorical_with_context(self):
        result = parse_categorical_response(
            "After careful consideration, I believe the answer is yes.",
            ["yes", "no"],
        )
        assert result == "yes"

    def test_parse_categorical_no_match(self):
        result = parse_categorical_response("maybe", ["yes", "no"])
        assert result is None

    def test_parse_categorical_multi_category(self):
        categories = ["positive", "negative", "neutral"]
        result = parse_categorical_response("This is positive", categories)
        assert result == "positive"

    def test_parse_categorical_true_as_yes(self):
        result = parse_categorical_response("true", ["yes", "no"])
        assert result == "yes"

    def test_parse_categorical_false_as_no(self):
        result = parse_categorical_response("false", ["yes", "no"])
        assert result == "no"


class TestGradedParsing:
    """Tests for graded response parsing."""

    def test_parse_graded_single_number(self):
        result = parse_graded_response("3", 1, 5)
        assert result == 3

    def test_parse_graded_with_context(self):
        result = parse_graded_response("I would rate this a 4 out of 5.", 1, 5)
        assert result == 4

    def test_parse_graded_out_of_range_high(self):
        result = parse_graded_response("10", 1, 5)
        assert result is None

    def test_parse_graded_out_of_range_low(self):
        result = parse_graded_response("0", 1, 5)
        assert result is None

    def test_parse_graded_first_valid_number(self):
        result = parse_graded_response("Rating: 7/10, but adjusted to 3", 1, 5)
        # Should find 3 which is in range
        assert result == 3

    def test_parse_graded_no_numbers(self):
        result = parse_graded_response("Excellent!", 1, 5)
        assert result is None

    def test_parse_graded_boundary_min(self):
        result = parse_graded_response("1", 1, 5)
        assert result == 1

    def test_parse_graded_boundary_max(self):
        result = parse_graded_response("5", 1, 5)
        assert result == 5

    def test_parse_graded_different_scale(self):
        result = parse_graded_response("Score: 75", 0, 100)
        assert result == 75


class TestThinkingBlockStripping:
    """Tests for thinking block removal."""

    def test_strip_thinking_basic(self):
        text = "<think>Internal reasoning here</think>Final answer: yes"
        result = strip_thinking_block(text)
        assert result == "Final answer: yes"

    def test_strip_thinking_multiline(self):
        text = """<think>
Line 1 of thinking
Line 2 of thinking
</think>
The answer is no."""
        result = strip_thinking_block(text)
        assert "think" not in result.lower()
        assert "The answer is no." in result

    def test_strip_thinking_multiple_blocks(self):
        text = "<think>First</think>Middle<think>Second</think>End"
        result = strip_thinking_block(text)
        assert result == "MiddleEnd"

    def test_strip_thinking_no_blocks(self):
        text = "No thinking blocks here"
        result = strip_thinking_block(text)
        assert result == "No thinking blocks here"

    def test_strip_thinking_empty_block(self):
        text = "<think></think>Content"
        result = strip_thinking_block(text)
        assert result == "Content"


class TestParsingWithThinkingBlock:
    """Tests for parsing responses that contain thinking blocks."""

    def test_categorical_with_thinking(self):
        text = "<think>Let me consider... the answer should be yes</think>yes"
        cleaned = strip_thinking_block(text)
        result = parse_categorical_response(cleaned, ["yes", "no"])
        assert result == "yes"

    def test_graded_with_thinking(self):
        text = "<think>Hmm, this deserves a 4 or maybe 5</think>I rate this 4."
        cleaned = strip_thinking_block(text)
        result = parse_graded_response(cleaned, 1, 5)
        assert result == 4

    def test_categorical_ignore_thinking_content(self):
        # The thinking block says "no" but the final answer is "yes"
        text = "<think>This might be no, but actually...</think>yes"
        cleaned = strip_thinking_block(text)
        result = parse_categorical_response(cleaned, ["yes", "no"])
        assert result == "yes"


class TestEdgeCases:
    """Tests for edge cases in parsing."""

    def test_empty_response(self):
        result = parse_categorical_response("", ["yes", "no"])
        assert result is None

    def test_whitespace_only(self):
        result = parse_categorical_response("   \n\t  ", ["yes", "no"])
        assert result is None

    def test_numeric_categories_as_strings(self):
        result = parse_categorical_response("Category 2", ["1", "2", "3"])
        assert result == "2"

    def test_mixed_case_categories(self):
        result = parse_categorical_response("POSITIVE", ["Positive", "Negative"])
        assert result == "Positive"
