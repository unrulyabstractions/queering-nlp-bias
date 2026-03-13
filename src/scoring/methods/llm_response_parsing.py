"""Shared utilities for parsing LLM judgment responses.

Provides common helpers for stripping thinking blocks and normalizing
responses from language model judges.
"""

from __future__ import annotations


def strip_thinking_content(response: str) -> str:
    """Strip thinking block content from model response.

    Removes content before the last </think> tag, which is used by
    reasoning models to show their thought process before the final answer.

    Args:
        response: Raw model response potentially containing thinking blocks

    Returns:
        Response text after any thinking blocks, stripped of whitespace
    """
    text = response
    if "</think>" in text:
        text = text.split("</think>")[-1]
    return text.strip()
