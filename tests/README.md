# Test Suite

Comprehensive test suite for the queering-nlp-bias pipeline.

## Running Tests

```bash
# Run all tests (excludes slow tests by default)
uv run pytest tests/

# Run ALL tests including slow ones
uv run pytest tests/ -m ""

# Run only unit tests
uv run pytest tests/unit/

# Run only integration tests
uv run pytest tests/integration/

# Run with verbose output
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/unit/common/test_base_schema.py

# Run specific test class
uv run pytest tests/unit/common/test_base_schema.py::TestBaseSchemaToDict

# Run with coverage report
uv run pytest tests/ --cov=src --cov-report=term-missing

# Run with HTML coverage report
uv run pytest tests/ --cov=src --cov-report=html
```

## Test Structure

```
tests/
├── conftest.py                 # Root fixtures and markers
├── fixtures/                   # Reusable test factories
│   ├── mock_model_runner.py    # MockModelRunner for testing without real models
│   ├── sample_trajectories.py  # TokenTrajectory factories
│   └── sample_compliance_data.py  # Compliance vector factories
│
├── unit/                       # Unit tests (fast, isolated)
│   ├── common/                 # Core data structures
│   │   ├── test_base_schema.py
│   │   ├── test_token_trajectory.py
│   │   ├── test_token_tree.py
│   │   ├── test_branching_node.py
│   │   └── math/               # Math utilities
│   │       ├── test_probability_utils.py
│   │       ├── test_structure_aware.py
│   │       └── test_numerical_edge_cases.py
│   ├── generation/
│   │   └── test_generation_config.py
│   ├── scoring/
│   │   ├── test_scoring_config.py
│   │   └── test_response_parsing.py
│   ├── estimation/
│   │   ├── test_weighting_methods.py
│   │   ├── test_arm_classification.py
│   │   └── test_core_computation.py
│   └── inference/
│       └── test_generated_trajectory.py
│
└── integration/                # Integration tests (full pipeline flows)
    ├── test_generation_pipeline.py
    ├── test_scoring_pipeline.py
    └── test_estimation_pipeline.py
```

## Test Markers

- `@pytest.mark.slow` - Tests that take a long time (excluded by default)
- `@pytest.mark.integration` - Integration tests

## Fixtures

### Mock Model Runner

```python
from fixtures.mock_model_runner import MockModelRunner, create_mock_runner

runner = create_mock_runner()
traj = runner.generate_trajectory_from_prompt("Test prompt", max_new_tokens=10)
```

### Sample Trajectories

```python
from fixtures.sample_trajectories import (
    create_sample_trajectory,
    create_divergent_trajectories,
    create_trajectory_batch,
)

# Single trajectory
traj = create_sample_trajectory(n_tokens=10, arm_idx=(0,))

# Trajectories that diverge at position 5
trajs = create_divergent_trajectories(n_trajectories=3, diverge_at=5)

# Batch with groups
batch = create_trajectory_batch(n_groups=2, trajs_per_group=3)
```

### Compliance Data

```python
from fixtures.sample_compliance_data import (
    create_compliance_vectors,
    create_trajectory_scoring_data,
    create_weights_for_trajectories,
)

vectors = create_compliance_vectors(n_samples=10, n_structures=4)
weights = create_weights_for_trajectories(10, weight_type="uniform")
```

## Writing New Tests

1. Place unit tests in `tests/unit/<module>/`
2. Place integration tests in `tests/integration/`
3. Use fixtures from `tests/fixtures/` for test data
4. Mark slow tests with `@pytest.mark.slow`
5. Mark integration tests with `@pytest.mark.integration`
