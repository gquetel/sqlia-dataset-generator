import logging
import os
import random
import re

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

logger = logging.getLogger(__name__)

# Auto-generated patterns by field type.
# {val} is replaced by the field's dict placeholder at generation time.
PATTERNS_BY_TYPE = {
    "string": [
        '= "{val}"',
        'LIKE "{val}"',
        'IN ("{val}","{val}")',
        'IN ("{val}","{val}","{val}")',
    ],
    "numeric": [
        "= {val}",
        ">= {val}",
        "<= {val}",
        "BETWEEN {val} AND {val}",
    ],
    "date": [
        '= "{val}"',
        '>= "{val}"',
        '<= "{val}"',
        'BETWEEN "{val}" AND "{val}"',
    ],
}


class ConditionGenerator:
    """Generate variable-condition WHERE clauses from a declarative conditions.toml file.

    Each dataset can define a conditions.toml with tables, fields, types and
    optional custom patterns. This class loads that config and generates random
    condition strings for normal query generation and attack template instantiation.
    """

    def __init__(self, dataset_path: str, fill_placeholder_fn: callable):
        """Load conditions.toml and prepare condition generation.

        Args:
            dataset_path: Path to the dataset directory (e.g. ./data/datasets/OurAirports)
            fill_placeholder_fn: Callable with signature (query, placeholder, count) -> (query, filler)
        """
        self.fill_placeholder_fn = fill_placeholder_fn
        self.tables = {}

        toml_path = os.path.join(dataset_path, "conditions.toml")
        if not os.path.exists(toml_path):
            logger.debug(f"No conditions.toml found at {toml_path}")
            return

        with open(toml_path, "rb") as f:
            config = tomllib.load(f)

        for table_def in config.get("table", []):
            name = table_def["name"]
            self.tables[name] = {
                "select_columns": table_def.get("select_columns", "*"),
                "min_conditions": table_def.get("min_conditions", 2),
                "max_conditions": table_def.get("max_conditions", 5),
                "fields": [],
            }
            for field_def in table_def.get("field", []):
                field = {
                    "column": field_def["column"],
                    "type": field_def["type"],
                }
                if field_def["type"] == "custom":
                    field["patterns"] = field_def["patterns"]
                else:
                    # Typed field: build patterns from column name and dict
                    field["dict"] = field_def["dict"]
                    field["patterns"] = self._build_typed_patterns(
                        field_def["column"], field_def["dict"], field_def["type"]
                    )
                self.tables[name]["fields"].append(field)

        logger.info(
            f"Loaded conditions for {len(self.tables)} table(s) from {toml_path}"
        )

    @staticmethod
    def _build_typed_patterns(
        column: str, dict_name: str, field_type: str
    ) -> list[str]:
        """Build condition patterns for a typed field.

        Replaces {val} in the type's pattern templates with the dict placeholder,
        and prepends the column name.

        Returns:
            List of pattern strings with {dict_name} placeholders.
        """
        base_patterns = PATTERNS_BY_TYPE[field_type]
        result = []
        for pattern in base_patterns:
            filled = pattern.replace("{val}", "{" + dict_name + "}")
            result.append(f"{column} {filled}")
        return result

    def has_conditions(self, table_name: str) -> bool:
        """Check if a table has condition definitions."""
        return table_name in self.tables

    def generate(self, table_name: str) -> tuple[str, list]:
        """Generate a random condition string with values filled.

        Used for normal query generation. Picks N random fields, picks a random
        pattern per field, fills all placeholders with values.

        Returns:
            Tuple of (condition_string, user_inputs_list)
        """
        unfilled = self.instantiate_template(table_name)

        # Now fill all placeholders in the instantiated condition string
        user_inputs = []
        all_placeholders = re.findall(r"\{([-a-zA-Z_]+)\}", unfilled)

        for placeholder in all_placeholders:
            unfilled, filler = self.fill_placeholder_fn(
                query=unfilled,
                placeholder=placeholder,
                count=1,
            )
            user_inputs.append(filler)

        return unfilled, user_inputs

    def instantiate_template(self, table_name: str) -> str:
        """Generate a condition string with dict placeholders kept unfilled.

        Used to create a concrete attack template from a {conditions} template.
        Picks N random fields and a random pattern per field, but does NOT fill
        the dict placeholders with actual values.

        Returns:
            Condition string with {dict_name} placeholders, e.g.:
            'type = "{airports_type}" AND elevation_ft >= {airports_elevation_ft}'
        """
        table = self.tables[table_name]
        n_conds = random.randint(table["min_conditions"], table["max_conditions"])
        chosen_fields = random.sample(table["fields"], n_conds)

        conditions = []
        for field in chosen_fields:
            pattern = random.choice(field["patterns"])
            conditions.append(pattern)

        return " AND ".join(conditions)
