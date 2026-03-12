# Logging Module

Centralized logging primitives for experiment pipelines.

## Contents

- `log_primitives.py` - Basic logging functions (`log`, `log_flush`, `log_progress`, `log_done`, `log_section`)
- `text_formatting.py` - Text formatting utilities (`center`, `pad_left`, `pad_right`, `indent`, `fmt_prob`, `fmt_core`, `oneline`, `preview`)
- `section_headers.py` - Section header formatting (`log_box`, `log_header`, `log_major`, `log_stage`, `log_step`, `log_divider`, `log_banner`, `log_sub_banner`, `log_section_title`, `log_pipeline_header`)
- `table_formatting.py` - Table formatting (`log_table_header`, `log_table_row`)
- `content_logging.py` - Content output utilities (`log_params`, `log_kv`, `log_items`, `log_wrapped`)
- `function_decorators.py` - Function decorators (`logged`)

## Usage

```python
from src.common.logging import log, log_header, log_step, fmt_prob

# Basic logging
log("Processing trajectories...")
log("Done!", gap=1)  # Blank line before message

# Section headers (with Unicode box-drawing characters)
log_header("Analysis Results")
log_step(1, "Load data", "from file")

# Format numbers
prob_str = fmt_prob(0.0001)  # "  1.0e-04"
```

## Constants

- `HEADER_WIDTH` = 60 (width for boxed headers)
- `BANNER_WIDTH` = 70 (width for banners)
- `STAGE_GAP` = 4 (blank lines before stage headers)
