# Logging Module

Centralized logging utilities with consistent formatting for console output and structured data display.

## Contents

- `log_primitives.py` - Core console output: `log()`, `log_flush()`, `log_progress()`, `log_done()`, `log_section()`
- `text_formatting.py` - Text alignment and formatting: `center()`, `pad_left()`, `pad_right()`, `indent()`, `fmt_prob()`, `fmt_core()`, `oneline()`
- `section_headers.py` - Header and banner formatting with Unicode box-drawing characters: `log_box()`, `log_header()`, `log_major()`, `log_stage()`, `log_step()`, `log_divider()`, `log_banner()`, `log_sub_banner()`, `log_section_title()`, `log_pipeline_header()`
- `table_formatting.py` - Table output: `log_table_header()`, `log_table_row()`
- `content_logging.py` - Structured data output: `log_params()`, `log_kv()`, `log_items()`, `log_wrapped()`
- `function_decorators.py` - Function call logging: `logged()`

## Usage Examples

```python
from src.common.logging import (
    log, log_header, log_step, fmt_prob,
    log_items, log_wrapped, logged
)

# Basic logging
log("Processing trajectories...")
log("Done!", gap=1)  # Add blank line before message

# Section headers
log_header("Analysis Results")
log_step(1, "Load data", "from file")

# Format numbers
prob_str = fmt_prob(0.0001)  # Returns "  1.0e-04"

# Structured content
log_items("Results:", ["item 1", "item 2"], prefix="r")
log_wrapped("Long text that spans multiple lines...")

# Decorate functions to log calls
@logged("my_function")
def process_data(param=None):
    pass
```

## Constants

- `HEADER_WIDTH` = 60 (width for boxed headers)
- `BANNER_WIDTH` = 70 (width for banners)
- `STAGE_GAP` = 4 (blank lines before stage headers)
