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
import shutil
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestLauncherIntegration:
    """Integration tests for launcher.py with actual config files"""

    @pytest.mark.parametrize(
        "dataset_name,attack_ratio_tolerance",
        [
            ("OurAirports", 0.01),
            ("OHR", 0.01),
            ("sakila", 0.01),
            ("AdventureWorks", 0.01),
        ],
    )
    @pytest.mark.slow
    @pytest.mark.integration
    def test_valid_single_dataset_config(
        self, tmp_path, subtests, dataset_name, attack_ratio_tolerance
    ):
        """Test launcher with valid single dataset configuration (parametrized for multiple datasets)"""
        # Use real dataset files from data/datasets/
        # Only output to tmp_path to avoid polluting the repo during tests.

        # Create output directory in tmp_path
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Create valid config file that outputs to tmp_path
        # Using same proportions as config.toml (7/10, 1/10, 1/10, 1/10)
        config_file = tmp_path / "test_config.toml"
        config_file.write_text(
            f"""
[general]
output_path = "dataset.csv"
attacks_ratio = 0.1
normal_only_template_ratio = 0.1
seed = 42

[mysql]
user = "tata"
password = "tata"
host = "localhost"
port = 61337
priv_user = "root"
priv_pwd = "root"

[[datasets]]
name = "{dataset_name}"

[datasets.statements]
select = "7/10"
delete = "1/10"
update = "1/10"
insert = "1/10"
"""
        )

        # Run launcher with full generation (testing mode uses a few templates only)
        result = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).parent.parent / "launcher.py"),
                "--config-file",
                str(config_file),
                "--testing",
                "--output-dir",
                str(output_dir),
            ],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent),  # Run from repo root
        )

        # Check successful generation
        combined_output = result.stdout + result.stderr

        with subtests.test(msg="success message appears in output"):
            assert (
                f"Dataset {dataset_name} saved successfully" in combined_output
            ), f"Expected success message not found. Output:\n{combined_output}"

        with subtests.test(msg="no critical errors in stderr"):
            assert (
                "CRITICAL" not in result.stderr
            ), f"CRITICAL error found in stderr:\n{result.stderr}"

        with subtests.test(msg="output file exists"):
            output_file = output_dir / f"{dataset_name}.csv"
            assert output_file.exists(), f"{dataset_name}.csv not found in {output_dir}"

        # Load and validate dataset contents
        df = pd.read_csv(output_dir / f"{dataset_name}.csv")

        with subtests.test(msg="dataset is not empty"):
            assert len(df) > 0, f"Generated {dataset_name} dataset is empty"

        with subtests.test(msg="attack samples present"):
            n_attacks = len(df[df["label"] == 1])
            assert n_attacks > 0, f"No attack samples found in {dataset_name} dataset"

        with subtests.test(msg="normal samples present"):
            n_normal = len(df[df["label"] == 0])
            assert n_normal > 0, f"No normal samples found in {dataset_name} dataset"

        with subtests.test(msg="attack ratio is correct"):
            n_attacks = len(df[df["label"] == 1])
            n_total = len(df)
            actual_ratio = n_attacks / n_total
            expected_ratio = 0.1
            assert (
                abs(actual_ratio - expected_ratio) < attack_ratio_tolerance
            ), f"Attack ratio {actual_ratio:.3f} differs from expected {expected_ratio} (tolerance: {attack_ratio_tolerance})"

        with subtests.test(msg="all attacks in test set"):
            attacks = df[df["label"] == 1]
            assert (
                attacks["split"] == "test"
            ).all(), f"Not all attack samples are in test set for {dataset_name}"

        with subtests.test(msg="insider attacks are present"):
            assert "attack_technique" in df.columns, "Missing attack_technique column"
            insider_attacks = df[df["attack_technique"] == "insider"]
            assert (
                len(insider_attacks) > 0
            ), f"No insider attacks found in {dataset_name} dataset"

        with subtests.test(msg="sqlmap-generated attacks are present"):
            # sqlmap attacks have attack_technique != "insider" and label == 1
            attacks = df[df["label"] == 1]
            sqlmap_attacks = attacks[attacks["attack_technique"] != "insider"]
            assert (
                len(sqlmap_attacks) > 0
            ), f"No sqlmap-generated attacks found in {dataset_name} dataset"

        with subtests.test(msg="normal samples in both train and test"):
            normal = df[df["label"] == 0]
            assert (
                "train" in normal["split"].values
            ), f"No normal samples in train set for {dataset_name}"
            assert (
                "test" in normal["split"].values
            ), f"No normal samples in test set for {dataset_name}"

        with subtests.test(
            msg="normal_only_template_ratio reserves templates correctly"
        ):
            assert (
                "query_template_id" in df.columns
            ), "Missing query_template_id column in generated dataset"

            # Get templates used for attacks
            attack_templates = set(df[df["label"] == 1]["query_template_id"].unique())

            # Get templates used only for normal samples
            normal_only_templates = (
                set(df[df["label"] == 0]["query_template_id"].unique())
                - attack_templates
            )

            assert (
                len(normal_only_templates) > 0
            ), f"No templates reserved for normal-only generation in {dataset_name}"

            # Verify that normal-only templates never appear in attack samples
            for template_id in normal_only_templates:
                attack_count = len(
                    df[(df["query_template_id"] == template_id) & (df["label"] == 1)]
                )
                assert (
                    attack_count == 0
                ), f"Template {template_id} marked as normal-only but found in {attack_count} attack samples"

        with subtests.test(msg="merged dataset file exists"):
            merged_file = output_dir / "dataset.csv"
            assert merged_file.exists(), f"Merged dataset.csv not found in {output_dir}"

        with subtests.test(
            msg="merged dataset contains same data as individual dataset"
        ):
            merged_df = pd.read_csv(output_dir / "dataset.csv")
            assert len(merged_df) == len(
                df
            ), f"Merged dataset has {len(merged_df)} samples but individual dataset has {len(df)} samples"
            assert (
                merged_df["label"] == df["label"]
            ).all(), "Merged dataset labels don't match individual dataset"

        with subtests.test(msg="dataset size relationships are correct"):
            # Count samples by split and label
            n_attacks = len(df[df["label"] == 1])
            n_train = len(df[df["split"] == "train"])
            n_test = len(df[df["split"] == "test"])

            # Check that attacks and train set are similar in size
            # Allow for some variance since exact equality may not be achievable
            size_ratio = n_attacks / n_train if n_train > 0 else float("inf")
            assert (
                0.9 <= size_ratio <= 1.1
            ), f"Attacks ({n_attacks}) and train set ({n_train}) sizes are not similar (ratio: {size_ratio:.2f}, expected ~1.0)"

            # Check that test set is approximately 10x the train set
            test_train_ratio = n_test / n_train if n_train > 0 else float("inf")
            assert (
                9.0 <= test_train_ratio <= 11.0
            ), f"Test set ({n_test}) should be ~10x train set ({n_train}), but ratio is {test_train_ratio:.2f}"

        with subtests.test(msg="multiple statement types are present"):
            assert (
                "statement_type" in df.columns
            ), "Missing statement_type column in generated dataset"

            statement_types = df["statement_type"].unique()
            expected_types = {"select", "delete", "update", "insert", "insider"}

            # Check that all expected statement types are present
            assert (
                set(statement_types) == expected_types
            ), f"Expected statement types {expected_types} but found {set(statement_types)}"

            # Verify approximate proportions (allowing for some variance in testing mode)
            total_samples = len(df)
            select_ratio = len(df[df["statement_type"] == "select"]) / total_samples
            delete_ratio = len(df[df["statement_type"] == "delete"]) / total_samples
            update_ratio = len(df[df["statement_type"] == "update"]) / total_samples
            insert_ratio = len(df[df["statement_type"] == "insert"]) / total_samples

            # Allow significant tolerance in testing mode since we use reduced templates
            assert (
                0.5 <= select_ratio <= 0.9
            ), f"SELECT proportion {select_ratio:.2f} is outside expected range [0.5, 0.9]"
            assert (
                0.05 <= delete_ratio <= 0.2
            ), f"DELETE proportion {delete_ratio:.2f} is outside expected range [0.05, 0.2]"
            assert (
                0.05 <= update_ratio <= 0.2
            ), f"UPDATE proportion {update_ratio:.2f} is outside expected range [0.05, 0.2]"
            assert (
                0.05 <= insert_ratio <= 0.2
            ), f"INSERT proportion {insert_ratio:.2f} is outside expected range [0.05, 0.2]"

    @pytest.mark.integration
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
        config_file.write_text(
            """
[general]
output_path = "dataset.csv"
attacks_ratio = 0.1
normal_only_template_ratio = 0.1
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
"""
        )

        monkeypatch.chdir(tmp_path)

        # Run launcher - should fail because there is no select.csv files
        result = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).parent.parent / "launcher.py"),
                "--config-file",
                str(config_file),
                "--testing",
            ],
            capture_output=True,
            text=True,
        )

        # Should fail with non-zero exit code
        assert (
            result.returncode != 0
        ), f"Expected failure but got success. Output:\n{result.stdout}\n{result.stderr}"

        # Should show error about missing statement CSV file
        combined_output = result.stdout + result.stderr
        assert (
            "Statement CSV file not found" in combined_output
        ), f"Expected error about missing CSV file. Output:\n{combined_output}"

        # Should mention Library
        assert (
            "Library" in combined_output
        ), f"Expected dataset name in error message. Output:\n{combined_output}"

    @pytest.mark.integration
    def test_invalid_missing_dataset_folder(self, tmp_path):
        """Test launcher fails when dataset folder is missing"""
        # Setup: OurAirports exists in real repo data dir, but NonExistentDataset doesn't
        # Create config with missing dataset
        config_file = tmp_path / "test_config.toml"
        config_file.write_text(
            """
[general]
output_path = "dataset.csv"
attacks_ratio = 0.1
normal_only_template_ratio = 0.1
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
"""
        )

        # Run launcher from repo root (don't change directory)
        result = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).parent.parent / "launcher.py"),
                "--config-file",
                str(config_file),
                "--testing",
            ],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent),  # Run from repo root
        )

        # Should fail with non-zero exit code
        assert result.returncode != 0
        # Should show error about missing dataset
        assert "NonExistentDataset" in result.stderr
        assert "Dataset folder not found" in result.stderr

    @pytest.mark.integration
    def test_invalid_empty_datasets(self, tmp_path, monkeypatch):
        """Test launcher fails when no datasets are configured"""
        # Setup: Create datasets directory but no datasets
        datasets_dir = tmp_path / "data" / "datasets"
        datasets_dir.mkdir(parents=True)

        # Create config with no datasets
        config_file = tmp_path / "test_config.toml"
        config_file.write_text(
            """
[general]
output_path = "dataset.csv"
attacks_ratio = 0.1
normal_only_template_ratio = 0.1
seed = 42

[mysql]
user = "tata"
password = "tata"
host = "localhost"
port = 61337
priv_user = "root"
priv_pwd = "root"
"""
        )

        monkeypatch.chdir(tmp_path)

        # Run launcher - should fail
        result = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).parent.parent / "launcher.py"),
                "--config-file",
                str(config_file),
                "--testing",
            ],
            capture_output=True,
            text=True,
        )

        # Should fail with non-zero exit code
        assert result.returncode != 0
        # Should show error about no datasets
        assert "No datasets configured" in result.stderr

    @pytest.mark.integration
    def test_invalid_missing_attacks_ratio(self, tmp_path, monkeypatch):
        """Test launcher fails when attacks_ratio is missing from config"""
        # Setup: Create minimal dataset structure
        datasets_dir = tmp_path / "data" / "datasets" / "TestDataset"
        datasets_dir.mkdir(parents=True)
        (datasets_dir / "queries").mkdir()
        (datasets_dir / "dicts").mkdir()

        # Create minimal CSV file
        (datasets_dir / "queries" / "select.csv").write_text(
            "template,ID,description\nSELECT 1,test-S1,Test query"
        )

        # Create config without attacks_ratio
        config_file = tmp_path / "test_config.toml"
        config_file.write_text(
            """
[general]
output_path = "dataset.csv"
normal_only_template_ratio = 0.1
seed = 42

[mysql]
user = "tata"
password = "tata"
host = "localhost"
port = 61337
priv_user = "root"
priv_pwd = "root"

[[datasets]]
name = "TestDataset"
[datasets.statements]
select = "1/1"
"""
        )

        monkeypatch.chdir(tmp_path)

        # Run launcher - should fail
        result = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).parent.parent / "launcher.py"),
                "--config-file",
                str(config_file),
                "--testing",
            ],
            capture_output=True,
            text=True,
        )

        # Should fail with non-zero exit code
        assert (
            result.returncode != 0
        ), f"Expected failure but got success. Output:\n{result.stdout}\n{result.stderr}"

        # Should show error about missing attacks_ratio
        combined_output = result.stdout + result.stderr
        assert (
            "attacks_ratio" in combined_output
        ), f"Expected error about missing attacks_ratio. Output:\n{combined_output}"
        assert (
            "Missing required" in combined_output or "ValueError" in combined_output
        ), f"Expected ValueError about missing parameter. Output:\n{combined_output}"

    @pytest.mark.integration
    def test_invalid_missing_normal_only_template_ratio(self, tmp_path):
        """Test launcher fails when normal_only_template_ratio is missing from config"""
        # Use real OurAirports data to simplify test - we only care about config validation
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Create config without normal_only_template_ratio
        config_file = tmp_path / "test_config.toml"
        config_file.write_text(
            """
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
"""
        )

        # Run launcher - should fail during config validation
        result = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).parent.parent / "launcher.py"),
                "--config-file",
                str(config_file),
                "--testing",
                "--output-dir",
                str(output_dir),
            ],
            capture_output=True,
            text=True,
        )

        # Should fail with non-zero exit code
        assert (
            result.returncode != 0
        ), f"Expected failure but got success. Output:\n{result.stdout}\n{result.stderr}"

        # Should show error about missing normal_only_template_ratio
        combined_output = result.stdout + result.stderr
        assert (
            "normal_only_template_ratio" in combined_output
        ), f"Expected error about missing normal_only_template_ratio. Output:\n{combined_output}"
        assert (
            "Missing required" in combined_output or "ValueError" in combined_output
        ), f"Expected ValueError about missing parameter. Output:\n{combined_output}"

    @pytest.mark.parametrize(
        "dataset_name,template_id,template_query",
        [
            (
                "OurAirports",
                "airport-S23",
                '"SELECT id FROM airport WHERE {conditions}",airport-S23,Retrieve airport IDs based on a flexible set of search conditions.,string',
            ),
            (
                "sakila",
                "sakila-S12",
                '"SELECT film_id FROM film WHERE {conditions}",sakila-S12,Search films using a flexible set of filter conditions.',
            ),
            (
                "AdventureWorks",
                "AW-S21",
                '"SELECT ProductID FROM Production_Product WHERE {conditions}",AW-S21,Search products using a flexible set of filter conditions.',
            ),
            (
                "OHR",
                "OHR-S14",
                '"SELECT employee_id FROM employees WHERE {conditions}",OHR-S14,Search employees using a flexible set of filter conditions.',
            ),
        ],
    )
    @pytest.mark.slow
    @pytest.mark.integration
    def test_conditions_template_generation(
        self, tmp_path, subtests, dataset_name, template_id, template_query
    ):
        """Test that {conditions} templates generate valid normal and attack samples for each dataset."""
        repo_root = Path(__file__).parent.parent
        real_dataset_dir = repo_root / "data" / "datasets" / dataset_name

        # Copy dataset directory to tmp_path
        dest_dataset_dir = tmp_path / "data" / "datasets" / dataset_name
        shutil.copytree(real_dataset_dir, dest_dataset_dir)

        # Determine the CSV header from the original file
        original_select = real_dataset_dir / "queries" / "select.csv"
        with open(original_select) as f:
            header = f.readline().strip()

        # Replace select.csv with only the {conditions} template
        select_csv = dest_dataset_dir / "queries" / "select.csv"
        select_csv.write_text(f"{header}\n{template_query}\n")

        # Remove other query CSV files so only select is available
        for csv_file in (dest_dataset_dir / "queries").glob("*.csv"):
            if csv_file.name != "select.csv":
                csv_file.unlink()

        # Create output directory
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Write minimal config with select-only statements
        config_file = tmp_path / "test_config.toml"
        config_file.write_text(
            f"""
[general]
output_path = "dataset.csv"
attacks_ratio = 0.1
normal_only_template_ratio = 0
seed = 42

[mysql]
user = "tata"
password = "tata"
host = "localhost"
port = 61337
priv_user = "root"
priv_pwd = "root"

[[datasets]]
name = "{dataset_name}"

[datasets.statements]
select = "1"
"""
        )

        # Run launcher in testing mode
        result = subprocess.run(
            [
                sys.executable,
                str(repo_root / "launcher.py"),
                "--config-file",
                str(config_file),
                "--testing",
                "--output-dir",
                str(output_dir),
            ],
            capture_output=True,
            text=True,
            cwd=str(tmp_path),
        )

        combined_output = result.stdout + result.stderr

        with subtests.test(msg="success"):
            assert (
                "CRITICAL" not in result.stderr
            ), f"CRITICAL error found:\n{result.stderr}"
            assert (
                f"Dataset {dataset_name} saved successfully" in combined_output
            ), f"Expected success message not found. Output:\n{combined_output}"

        output_file = output_dir / f"{dataset_name}.csv"

        with subtests.test(msg="output exists"):
            assert output_file.exists(), f"{dataset_name}.csv not found in {output_dir}"

        if not output_file.exists():
            return  # Cannot validate further without output

        df = pd.read_csv(output_file)

        with subtests.test(msg="normal samples present"):
            n_normal = len(df[df["label"] == 0])
            assert n_normal > 0, f"No normal samples found in {dataset_name} dataset"

        with subtests.test(msg="attack samples present"):
            n_attacks = len(df[df["label"] == 1])
            assert n_attacks > 0, f"No attack samples found in {dataset_name} dataset"

        with subtests.test(msg="all queries use conditions template"):
            # Insider threat attacks have NaN template IDs; exclude them
            non_insider = df[df["attack_technique"] != "insider"]
            template_ids = non_insider["query_template_id"].dropna().unique()
            assert list(template_ids) == [
                template_id
            ], f"Expected only template {template_id} but found {list(template_ids)}"

        with subtests.test(msg="normal queries have WHERE clause with AND"):
            normal_queries = df[df["label"] == 0]["full_query"].tolist()
            queries_with_and = [q for q in normal_queries if " AND " in q]
            assert len(queries_with_and) > 0, (
                f"No normal queries contain ' AND ' (multiple conditions). "
                f"Sample queries: {normal_queries[:3]}"
            )
