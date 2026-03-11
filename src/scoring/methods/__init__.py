"""Scoring methods package.

Each scoring method is implemented in its own module:
- categorical: Binary yes/no judgments
- graded: 0-1 scale judgments
- similarity: Embedding cosine similarity
- count_occurrences: Word/phrase occurrence frequency
"""

from src.common.auto_export import auto_export

__all__ = auto_export(__file__, __name__, globals())
