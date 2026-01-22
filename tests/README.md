# Tests

This directory contains the test suite for the SQL injection dataset generator.

## Running Tests

### Run all tests

```bash
pytest
```

### Run specific test file

```bash
pytest tests/test_launcher_integration.py
```

### Run specific test

```bash
pytest tests/test_launcher_integration.py::TestLauncherIntegration::test_valid_single_dataset_config
```

### Run with verbose output

```bash
pytest -v
```

### Run with coverage

```bash
pytest --cov=src --cov-report=html
```

## Test Structure

- `test_launcher_integration.py` - Integration tests for launcher.py with actual TOML configuration files
  - Tests launcher invocation with valid configurations
  - Tests error handling for missing dataset folders
  - Tests error messages and exit codes
  - Uses subprocess to invoke launcher.py as a real user would

## Test Approach

The tests use an **integration testing** approach rather than unit testing:

- Creates temporary TOML configuration files
- Invokes `launcher.py` as a subprocess
- Validates exit codes and error messages
- Tests the complete validation flow from config parsing to dataset validation

This approach is more realistic as it tests the actual user-facing behavior rather than isolated functions.

## Writing New Tests

1. Create test files with `test_*.py` naming pattern
1. Use `Test*` class naming for test classes
1. Use `test_*` function naming for test methods
1. Use pytest fixtures (`tmp_path`, `monkeypatch`) for isolation
1. Mark slow tests with `@pytest.mark.slow`

## Fixtures

The test suite uses pytest's built-in fixtures:

- `tmp_path` - Provides a temporary directory for each test
- `monkeypatch` - Allows modifying environment, changing working directory, etc.

## Test Categories

Tests can be marked with categories:

- `@pytest.mark.unit` - Unit tests (fast, isolated)
- `@pytest.mark.integration` - Integration tests (slower, may require external services)
- `@pytest.mark.slow` - Slow tests (can be skipped with `-m "not slow"`)
