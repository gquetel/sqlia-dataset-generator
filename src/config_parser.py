from fractions import Fraction


def get_mysql_info(config: dict):
    mysql = config["mysql"]
    user = mysql["user"]
    pwd = mysql["password"]
    host = mysql["host"]
    port = mysql["port"]
    priv_user = mysql["priv_user"]
    priv_pwd = mysql["priv_pwd"]
    return user, pwd, host, port, priv_user, priv_pwd

def get_seed(config: dict):
    return config["general"]["seed"]

def get_attacks_ratio(config: dict):
    return config["general"]["attacks_ratio"]

def get_output_path(config: dict):
    return config["general"]["output_path"]

def get_used_datasets(config: dict):
    return [dataset["name"] for dataset in config.get("datasets", [])]

def get_statement_types_and_proportions(dataset_config: dict):
    """Extract statement types and their proportions from a dataset configuration.

    Args:
        dataset_config: A single dataset configuration dict from the TOML config

    Returns:
        List of dicts with 'type' and 'proportion' keys
    """
    stmts = []

    statements = dataset_config.get("statements", {})

    for stmt_type, proportion_str in statements.items():
        stmts.append({
            "type": stmt_type,
            "proportion": float(Fraction(proportion_str))
        })

    if abs(sum([stmt["proportion"] for stmt in stmts]) - 1.0) > 1e-10:
        raise ValueError(
            f"Proportions of queries types must sum up to 1. Current is {sum([stmt['proportion'] for stmt in stmts])}"
        )

    return stmts

def get_dataset_output_path(dataset_config: dict, general_config: dict):
    """Get the output path for a dataset, with fallback to general config."""
    return dataset_config.get("output_path", general_config.get("output_path", "dataset.csv"))

def get_dataset_attacks_ratio(dataset_config: dict, general_config: dict):
    """Get the attacks ratio for a dataset, with fallback to general config."""
    return dataset_config.get("attacks_ratio", general_config.get("attacks_ratio", 0.1))

def get_dataset_database_name(dataset_config: dict):
    """Get the database name for a dataset.

    Uses the dataset name as the database name.
    """
    return dataset_config.get("name", "dataset")
