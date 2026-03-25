"""
Base I/O utilities for saving and loading data.

Provides core JSON/JSONL utilities. Output-specific functions are in scripts/io.py.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path


def parse_file_path(filename, default_ext=".json", default_dir_path=""):
    """Parse filename into full file path.

    Args:
        filename: Simple name, file name, or full path
        default_ext: Extension to add if filename is a simple name
        default_dir_path: Directory to prepend if filename is not a full path

    Returns:
        Full file path as Path object
    """
    default_ext = default_ext if default_ext.startswith(".") else f".{default_ext}"
    if is_simple_name(filename):
        return Path(default_dir_path) / f"{filename}{default_ext}"
    elif is_file_name(filename):
        return Path(default_dir_path) / filename
    elif is_file_path(filename):
        return Path(filename)
    else:
        raise Exception(f"{filename} is not valid")


def is_simple_name(s: str) -> bool:
    """Check if string is a simple name (no path separators or extensions)."""
    return "/" not in s and "\\" not in s and "." not in s


def is_path(s: str) -> bool:
    """Check if string is a simple name (no path separators or extensions)."""
    return "/" in s or "\\" in s


def is_file_name(s: str, ext: str | None = None) -> bool:
    """Check if string looks like a file path (has extension or path separators).

    Args:
        s: String to check
        ext: Optional extension to match (e.g., ".json" or "json")
    """
    if is_simple_name(s):
        return False
    if is_path(s):
        return False
    if s.count(".") != 1:
        return False
    if ext:
        ext = ext if ext.startswith(".") else f".{ext}"
        if not s.endswith(ext):
            return False
    return True


def is_file_path(s: str, ext: str | None = None) -> bool:
    """Check if string looks like a file path (has extension or path separators).

    Args:
        s: String to check
        ext: Optional extension to match (e.g., ".json" or "json")
    """
    is_path = ("/" in s or "\\" in s or "." in s) and not s.endswith("/")
    if not is_path or ext is None:
        return is_path
    ext = ext if ext.startswith(".") else f".{ext}"
    return s.endswith(ext)


def get_timestamp() -> str:
    """Get current timestamp string."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: Path) -> Path:
    """Ensure directory exists."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def _make_text_readable(obj):
    """Recursively convert long text fields to arrays of lines for readability."""
    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            if k in ("text", "raw_text", "trace") and isinstance(v, str) and "\n" in v:
                # Convert multiline text to array of lines
                result[k] = v.split("\n")
            else:
                result[k] = _make_text_readable(v)
        return result
    elif isinstance(obj, list):
        return [_make_text_readable(item) for item in obj]
    else:
        return obj


def save_json(data, path: Path, readable_text: bool = True) -> None:
    """Save dictionary as pretty JSON."""
    if readable_text:
        data = _make_text_readable(data)
    with open(path, "w") as f:
        json.dump(data, f, indent=4, default=str, ensure_ascii=False)


def _restore_text_fields(obj):
    """Recursively restore text fields from arrays back to strings."""
    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            if k in ("text", "raw_text", "trace") and isinstance(v, list):
                # Join array of lines back to string
                result[k] = "\n".join(v)
            else:
                result[k] = _restore_text_fields(v)
        return result
    elif isinstance(obj, list):
        return [_restore_text_fields(item) for item in obj]
    else:
        return obj


def load_json(path: Path, default: dict | list | None = None) -> dict | list:
    """Load JSON file with extensive error recovery.

    Handles:
    - Empty files (returns default or raises)
    - Trailing/double commas
    - Truncated JSON (attempts repair)
    - BOM markers
    - Various encoding issues

    Args:
        path: Path to JSON file
        default: Default value if file is empty/missing (None = raise error)

    Returns:
        Parsed JSON data with text fields restored
    """
    path = Path(path)

    # Check file exists
    if not path.exists():
        if default is not None:
            return default
        raise FileNotFoundError(f"JSON file not found: {path}")

    # Read file content
    try:
        with open(path, encoding="utf-8") as f:
            s = f.read()
    except UnicodeDecodeError:
        # Try with latin-1 as fallback
        with open(path, encoding="latin-1") as f:
            s = f.read()

    # Handle empty file
    s = s.strip()
    if not s:
        if default is not None:
            return default
        raise ValueError(f"Empty JSON file: {path}")

    # Remove BOM if present
    if s.startswith("\ufeff"):
        s = s[1:]

    # Pre-processing: fix common JSON issues
    # Remove double/multiple commas (e.g., "a",, "b" -> "a", "b")
    s = re.sub(r",(\s*,)+", ",", s)
    # Remove trailing commas before ] or }
    s = re.sub(r",\s*([}\]])", r"\1", s)
    # Remove leading commas after [ or {
    s = re.sub(r"([{\[])(\s*),", r"\1\2", s)

    # Try to parse
    try:
        data = json.loads(s)
        return _restore_text_fields(data)
    except json.JSONDecodeError as e:
        # Attempt repair for truncated JSON
        repaired = _attempt_json_repair(s)
        if repaired is not None:
            try:
                data = json.loads(repaired)
                print(f"  [Warning] Repaired truncated JSON: {path}")
                return _restore_text_fields(data)
            except json.JSONDecodeError:
                pass

        # If we have a default, use it
        if default is not None:
            print(f"  [Warning] Failed to parse JSON, using default: {path}")
            return default

        # Provide helpful error message
        raise ValueError(
            f"Invalid JSON in {path} at line {e.lineno}, col {e.colno}: {e.msg}\n"
            f"Context: ...{s[max(0, e.pos - 30):e.pos + 30]}..."
        ) from e


def _attempt_json_repair(s: str) -> str | None:
    """Attempt to repair truncated/malformed JSON.

    Returns repaired string or None if repair not possible.
    """
    s = s.strip()
    if not s:
        return None

    # Count brackets to detect truncation
    open_braces = s.count("{") - s.count("}")
    open_brackets = s.count("[") - s.count("]")

    # If balanced, no repair needed (error is elsewhere)
    if open_braces == 0 and open_brackets == 0:
        return None

    repaired = s

    # Handle incomplete string at end (unclosed quote)
    # Count quotes - if odd, we have an unclosed string
    quote_count = repaired.count('"') - repaired.count('\\"')
    if quote_count % 2 == 1:
        repaired = repaired + '"'

    # Handle trailing colon (incomplete key-value pair)
    if re.search(r':\s*$', repaired):
        repaired = repaired + 'null'

    # Handle trailing comma
    repaired = re.sub(r',\s*$', '', repaired)

    # Recount after fixes
    open_braces = repaired.count("{") - repaired.count("}")
    open_brackets = repaired.count("[") - repaired.count("]")

    # Add missing closing brackets/braces in correct order
    # We need to close them in reverse order of opening
    # Simple heuristic: find last unmatched opener and close that first
    closings = []
    depth_brace = 0
    depth_bracket = 0

    for char in repaired:
        if char == "{":
            depth_brace += 1
            closings.append("}")
        elif char == "}":
            depth_brace -= 1
            if closings and closings[-1] == "}":
                closings.pop()
        elif char == "[":
            depth_bracket += 1
            closings.append("]")
        elif char == "]":
            depth_bracket -= 1
            if closings and closings[-1] == "]":
                closings.pop()

    # Reverse to get correct closing order
    repaired += "".join(reversed(closings))

    return repaired
