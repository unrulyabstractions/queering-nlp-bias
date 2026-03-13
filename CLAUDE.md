# Project Guidelines

## Running Scripts

**ALWAYS use `uv run` to execute Python scripts.** This project uses `uv` for dependency management. Never use bare `python` or `python3` commands.

```bash
# Correct:
uv run python scripts/generate_by_simple_sampling.py trials/generation/example.json

# Wrong:
python scripts/generate_by_simple_sampling.py trials/generation/example.json
```

## Critical Rules (NEVER violate these)

1. **BEFORE creating ANY function, EXHAUSTIVELY search for existing implementations.** Search `src/common/` and related modules for similar functionality. Only create a new function if you can justify why existing code cannot be reused. Common places to check:
   - `src/common/math/` - vector operations, norms, distances, entropy, probability
   - `src/common/text/` - string processing, EOS handling
   - `src/common/file_io.py` - JSON loading, path utilities
   - `src/common/device_utils.py` - memory, GPU/MPS operations

2. **ALL `__init__.py` files MUST use auto-export.** Every `__init__.py` automatically exports all public symbols from its submodules. No manual export lists. Use a pattern like:
   ```python
   # Auto-export all public symbols from submodules
   import importlib
   import pkgutil

   for _loader, _name, _is_pkg in pkgutil.iter_modules(__path__):
       _module = importlib.import_module(f".{_name}", __name__)
       for _attr in getattr(_module, "__all__", [k for k in dir(_module) if not k.startswith("_")]):
           globals()[_attr] = getattr(_module, _attr)
   ```

3. **ALL imports go at the top of every file.** No inline imports, no imports inside functions, no imports inside `if` blocks. The only exception is circular dependency resolution — and even then, prefer restructuring the code to eliminate the cycle.

4. **NO legacy code. NO backwards compatibility.** Remove all deprecated functions, shims, compatibility layers, and dead code paths. If something is replaced, delete the old version entirely.

5. **NO two `.py` files in the entire repo may share the same filename.** Every `.py` file must have a unique name across all directories.

6. **No single-word `.py` filenames.** Every `.py` file (except `__init__.py`) must be at least two words or a descriptive compound name. Bad: `base.py`, `utils.py`, `helpers.py`, `config.py`. Good: `generation_method_base.py`, `string_utils.py`, `scoring_config_loader.py`, `network_request_helpers.py`.

7. **No deep nesting in function signatures.** Nothing deeper than a 1D `dict` or `list` should be accepted as an argument or returned from a function. If you need nested structures, define a proper `BaseSchema` subclass instead.

8. **Always update documentation when modifying code.** When you modify `.py` files in a folder, update the corresponding `.md` files in that same folder (e.g., `README.md`, `EXPLANATION.md`). Keep documentation in sync with the code it describes.

## Code Quality Standards

### Clean Code
- No dead code, no commented-out code, no debug prints, no TODOs left behind
- No duplicate code — extract common patterns into shared utility functions
- Every function should do one thing and do it well
- Keep files small and focused — if a file exceeds ~150 lines, consider splitting it

### Abstraction & Reuse
- Use maximum abstraction: business logic lives in reusable subfunctions, not in top-level orchestration code
- Implementation details belong in small, reusable, well-named helper functions
- Top-level functions should read like a high-level description of what happens, delegating all work to subfunctions
- Never copy-paste logic — if two places need the same behavior, extract it

### Readability
- Code should be well-commented: explain *why*, not *what*
- Use clear, descriptive names for everything (functions, variables, classes, files)
- Keep functions short and focused
- Prefer explicit over implicit

## Architecture Patterns

### BaseSchema for All Data Classes
- **Every dataclass that holds structured data MUST inherit from `BaseSchema`** (located in `src/common/base_schema.py`)
- `BaseSchema` provides automatic `.to_dict()`, `.from_dict()`, and serialization support
- This applies to:
  - All analysis result dataclasses
  - All request/response objects
  - All configuration objects
  - Any dataclass that crosses a module boundary or gets serialized
- Never use raw `dict` for structured data — define a `BaseSchema` subclass instead
- Never accept or return nested `dict`/`list` structures — use typed `BaseSchema` fields

### Module Structure
- Keep `.py` files small (aim for under 150 lines)
- Each file should have a single clear responsibility
- Group related files into packages with auto-exporting `__init__.py`
- Prefer many small files over few large files

### Naming Conventions for Files
- Use descriptive, multi-word filenames: `analysis_runner.py`, `token_counter.py`, `schema_validator.py`
- Package directories can be single words: `analysis/`, `models/`, `utils/`
- Filenames must be globally unique across the entire repository

## What to Do When Refactoring

1. **Search for existing utilities FIRST** — before writing any new function, grep `src/common/` for similar functionality
2. **Check all `__init__.py` files** — ensure they use auto-export
3. **Check all imports** — move any inline/conditional imports to the top of the file
4. **Check for duplicate filenames** — rename any `.py` files that share a name with another file anywhere in the repo
5. **Check for single-word filenames** — rename them to be descriptive and multi-word
6. **Check for nested dict/list args** — replace with `BaseSchema` subclasses
7. **Remove all legacy/compat code** — delete deprecated functions, old API shims, backwards compatibility wrappers
8. **Extract repeated logic** — find duplicated code and move it into shared utilities
9. **Break up large files** — split any file over ~150 lines into smaller, focused modules