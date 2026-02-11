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


def get_output_path(config: dict):
    return config["general"]["output_path"]


def get_used_datasets(config: dict):
    return [dataset["name"] for dataset in config.get("datasets", [])]


def get_dataset_port(config: dict, dataset_name: str) -> int:
    """Return the MySQL port for a specific dataset (base port + offset)."""
    base_port = config["mysql"]["port"]
    for dataset in config.get("datasets", []):
        if dataset["name"] == dataset_name:
            return base_port + dataset.get("port_offset", 0)
    raise ValueError(f"Dataset '{dataset_name}' not found in configuration")


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
        stmts.append({"type": stmt_type, "proportion": float(Fraction(proportion_str))})

    if abs(sum([stmt["proportion"] for stmt in stmts]) - 1.0) > 1e-10:
        raise ValueError(
            f"Proportions of queries types must sum up to 1. Current is {sum([stmt['proportion'] for stmt in stmts])}"
        )

    return stmts


def get_attacks_ratio(config: dict):
    """Get the attacks ratio from general config."""
    general_config = config["general"]
    if "attacks_ratio" not in general_config:
        raise ValueError("Missing required 'attacks_ratio' in general config")
    return general_config["attacks_ratio"]


def get_normal_only_template_ratio(config: dict):
    """Get the normal-only template ratio from general config."""
    general_config = config["general"]
    if "normal_only_template_ratio" not in general_config:
        raise ValueError(
            "Missing required 'normal_only_template_ratio' in general config"
        )
    return general_config["normal_only_template_ratio"]
