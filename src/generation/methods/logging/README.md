# Generation Method Logging

Logging utilities for generation methods.

## Modules

### `gen_logging_utils.py`

Shared logging utilities for all generation methods.

**Functions:**

- `log_arm_header(arm, log_fn)` - Log arm name and prefill text
- `log_tree_trajectories(result, runner)` - Log trajectory texts and conditional probabilities with detailed formatting
  - Displays trajectories with prefill and generated text
  - Shows conditional probabilities (p(t|root), p(t|trunk), p(t|branch), p(t|twig))
  - Indicates whether each trajectory ends with EOS token
  - Uses stored trajectory fields for display without recomputation

### `entropy_seeking_logging.py`

Logging for entropy-seeking generation method.

**Functions:**

- `log_initialize_tree(tree_paths, runner, samples, max_tokens)` - Log tree initialization with ASCII visualization
- `log_expansion_round(...)` - Log a single expansion round with branch details
- `log_expansion_summary(tree_paths, initial_count, rounds)` - Log final path count summary
- `log_arm_header_entropy(arm_name, continuation)` - Log arm header for entropy-seeking output
- `log_tree_visualization(tree_paths, max_tokens)` - ASCII tree visualization with parent relationships
- `log_paths(tree_paths, runner, max_display)` - Display path continuations (first N paths)

### `forking_paths_logging.py`

Logging for forking paths generation method.

**Functions:**

- `log_greedy_path(greedy_traj, runner, arm_name, prompt_len)` - Log the greedy baseline path
- `log_position_analyses(analyses, qualifying, runner, params, greedy_traj, prompt_len)` - Log entropy analysis
  - Shows prompt and response tokens
  - Displays entropy statistics (min, max, mean, std, percentiles)
  - Shows entropy distribution histogram with dynamic binning
  - ASCII visualization of entropy across positions
  - Lists qualifying forks meeting probability threshold
- `log_fork_expansion(fork_points, analyses, runner, prompt_len)` - Log fork point expansion details
- `log_arm_tree(arm_name, greedy_traj, fork_points, runner, prompt_len, max_tokens)` - ASCII tree visualization for forking paths
- `_log_entropy_histogram(entropies, threshold)` - Internal histogram generation
- `_log_entropy_ascii(analyses, threshold)` - Internal entropy line chart visualization
