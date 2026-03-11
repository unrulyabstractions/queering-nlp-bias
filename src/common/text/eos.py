"""EOS token handling utilities.

This module provides utilities for handling end-of-sequence tokens
from various language models.

The eos_token MUST be obtained from the model via ModelRunner.eos_token
and passed to strip_eos_tokens(). There is no fallback - if no token
is provided, the text is returned unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence


def strip_eos_tokens(
    text: str,
    markers: Sequence[str] | None = None,
) -> str:
    """Remove EOS tokens from end of text.

    Strips trailing whitespace, then removes any matching EOS marker,
    then strips trailing whitespace again.

    Args:
        text: Text that may end with EOS markers
        markers: List of EOS markers to strip. Get this from ModelRunner.eos_token.
                 If None or empty, text is returned with only whitespace stripped.

    Returns:
        Text with EOS markers stripped from the end.

    Example:
        >>> strip_eos_tokens("Hello world<|im_end|>", ["<|im_end|>"])
        'Hello world'
    """
    result = text.rstrip()

    if not markers:
        return result

    for marker in markers:
        if marker and result.endswith(marker):
            result = result[: -len(marker)].rstrip()
            # Only strip one marker (they shouldn't stack)
            break

    return result
