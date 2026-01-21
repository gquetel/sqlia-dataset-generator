"""
Integration tests for launcher.py with configuration files
Generated using Claude Code.

IMPORTANT TESTING CONVENTIONS:

1. Using Real vs. Temporary Dataset Directories:

   - USE REAL REPO DATA (data/datasets/OurAirports):
     * When testing valid configurations that need actual dataset files
     * When testing that references an existing dataset (e.g., OurAirports exists, but another doesn't)
     * Set cwd=repo_root in subprocess.run() to ensure launcher finds data/datasets
     * Use --output-dir with tmp_path to avoid polluting repo with generated files

   - USE tmp_path FOR TEST DATA:
     * When testing validation errors for missing/invalid dataset structures
     * When you need to create isolated, incomplete dataset folders
     * Use monkeypatch.chdir(tmp_path) to make launcher look in tmp_path/data/datasets
     * Example: Testing missing CSV files, empty datasets, malformed structures

2. Working Directory Management:

   - launcher.py uses relative path "data/datasets" to find datasets
   - Either run from repo root (cwd=repo_root) OR change to tmp_path (monkeypatch.chdir)
   - Never mix both - this causes "Dataset directory not found" errors

3. Examples:

   Good: Test valid config with OurAirports
   → Use real data/datasets/OurAirports, set cwd=repo_root, output to tmp_path

   Good: Test OurAirports exists but NonExistentDataset doesn't
   → Use real data/datasets, set cwd=repo_root (OurAirports validates, NonExistentDataset fails)

   Good: Test missing CSV files in custom dataset
   → Create tmp_path/data/datasets/MyDataset with empty queries/, use monkeypatch.chdir(tmp_path)
"""

import pytest
from pathlib import Path
import sys
import subprocess

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestLauncherIntegration:
    """Integration tests for launcher.py with actual config files"""

    def test_valid_single_dataset_config(self, tmp_path):
        """Test launcher with valid single dataset configuration"""
        # Use real dataset files from data/datasets/OurAirports
        # Only output to tmp_path to avoid polluting the repo during tests.

        # Create output directory in tmp_path
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Create valid config file that outputs to tmp_path
        config_file = tmp_path / "test_config.toml"
        config_file.write_text(f"""
[general]
output_path = "dataset.csv"
attacks_ratio = 0.1
seed = 42

[mysql]
user = "tata"
password = "tata"
host = "localhost"
port = 61337
priv_user = "root"
priv_pwd = "root"

[[datasets]]
name = "OurAirports"

[datasets.statements]
select = "1/1"
""")

        # Run launcher with full generation (testing mode uses a few templates only)
        result = subprocess.run(
            [sys.executable, str(Path(__file__).parent.parent / "launcher.py"),
             "--config-file", str(config_file), "--testing", "--output-dir", str(output_dir)],
            capture_output=True,
            text=True
        )

        # Check successful generation
        combined_output = result.stdout + result.stderr

        assert "Dataset OurAirports saved successfully" in combined_output, \
            f"Expected success message not found. Output:\n{combined_output}"

        # Should not have CRITICAL errors
        assert "CRITICAL" not in result.stderr, \
            f"CRITICAL error found in stderr:\n{result.stderr}"

        # Check output files exist
        assert (output_dir / "OurAirports.csv").exists(), \
            f"OurAirports.csv not found in {output_dir}"

    def test_invalid_missing_statement_csv_files(self, tmp_path, monkeypatch):
        """Test launcher fails when statement CSV files are missing for datasets"""
        # Setup: Folders exists, but not the statement files. 
        dataset_name = "Library"
        datasets_dir = tmp_path / "data" / "datasets" / dataset_name
        datasets_dir.mkdir(parents=True)
        (datasets_dir / "queries").mkdir()
        (datasets_dir / "dicts").mkdir()

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Create config file with statements but their CSV files does not exists.  
        config_file = tmp_path / "test_config.toml"
        config_file.write_text("""
[general]
output_path = "dataset.csv"
attacks_ratio = 0.1
seed = 42

[mysql]
user = "tata"
password = "tata"
host = "localhost"
port = 61337
priv_user = "root"
priv_pwd = "root"

[[datasets]]
name = "Library"
[datasets.statements]
select = "1/1"
""")

        monkeypatch.chdir(tmp_path)

        # Run launcher - should fail because there is no select.csv files
        result = subprocess.run(
            [sys.executable, str(Path(__file__).parent.parent / "launcher.py"),
             "--config-file", str(config_file), "--testing"],
            capture_output=True,
            text=True
        )

        # Should fail with non-zero exit code
        assert result.returncode != 0, f"Expected failure but got success. Output:\n{result.stdout}\n{result.stderr}"

        # Should show error about missing statement CSV file
        combined_output = result.stdout + result.stderr
        assert "Statement CSV file not found" in combined_output, \
            f"Expected error about missing CSV file. Output:\n{combined_output}"

        # Should mention Library
        assert "Library" in combined_output, \
            f"Expected dataset name in error message. Output:\n{combined_output}"

    def test_invalid_missing_dataset_folder(self, tmp_path):
        """Test launcher fails when dataset folder is missing"""
        # Setup: OurAirports exists in real repo data dir, but NonExistentDataset doesn't
        # Create config with missing dataset
        config_file = tmp_path / "test_config.toml"
        config_file.write_text("""
[general]
output_path = "dataset.csv"
attacks_ratio = 0.1
seed = 42

[mysql]
user = "tata"
password = "tata"
host = "localhost"
port = 61337
priv_user = "root"
priv_pwd = "root"

[[datasets]]
name = "OurAirports"
[datasets.statements]
select = "1/1"

[[datasets]]
name = "NonExistentDataset"
[datasets.statements]
select = "1/1"
""")

        # Run launcher from repo root (don't change directory)
        result = subprocess.run(
            [sys.executable, str(Path(__file__).parent.parent / "launcher.py"),
             "--config-file", str(config_file), "--testing"],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent)  # Run from repo root
        )

        # Should fail with non-zero exit code
        assert result.returncode != 0
        # Should show error about missing dataset
        assert "NonExistentDataset" in result.stderr
        assert "Dataset folder not found" in result.stderr

    def test_invalid_empty_datasets(self, tmp_path, monkeypatch):
        """Test launcher fails when no datasets are configured"""
        # Setup: Create datasets directory but no datasets
        datasets_dir = tmp_path / "data" / "datasets"
        datasets_dir.mkdir(parents=True)

        # Create config with no datasets
        config_file = tmp_path / "test_config.toml"
        config_file.write_text("""
[general]
output_path = "dataset.csv"
attacks_ratio = 0.1
seed = 42

[mysql]
user = "tata"
password = "tata"
host = "localhost"
port = 61337
priv_user = "root"
priv_pwd = "root"
""")

        monkeypatch.chdir(tmp_path)

        # Run launcher - should fail
        result = subprocess.run(
            [sys.executable, str(Path(__file__).parent.parent / "launcher.py"),
             "--config-file", str(config_file), "--testing"],
            capture_output=True,
            text=True
        )

        # Should fail with non-zero exit code
        assert result.returncode != 0
        # Should show error about no datasets
        assert "No datasets configured" in result.stderr