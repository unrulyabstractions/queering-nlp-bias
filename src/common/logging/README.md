# Logging Module

> **Note**: This documentation was AI-generated and may contain errors. If something seems off, check the code or open an issue.


Core logging primitives for experiment pipelines.

## Contents

- `core.py` - Basic logging functions (`log`, `log_flush`)
- `formatting.py` - Output formatting utilities
- `headers.py` - Section header formatting
- `tables.py` - Table formatting utilities

## Usage

```python
from src.common.logging import log, log_flush

log("Processing trajectories...")
log("Done!", gap=1)  # Blank line before message
log_flush()          # Flush stdout
```

See also `src/common/log_utils.py` for higher-level logging utilities (headers, tables, banners).
