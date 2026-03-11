# Text Utilities

> **Note**: This documentation was AI-generated and may contain errors. If something seems off, check the code or open an issue.


Text processing utilities for trajectory analysis.

## Contents

- `display.py` - Display name formatting (arm names, structure labels)
- `eos.py` - End-of-sequence token handling

## Usage

```python
from src.common.text import arm_display_name, structure_label

# Arm display names
arm_display_name(0)   # "trunk"
arm_display_name(1)   # "branch_1"

# Structure labels (1-indexed)
structure_label(0, "c")  # "c1" (categorical)
structure_label(2, "g")  # "g3" (graded)
```
