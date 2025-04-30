import configparser
from fractions import Fraction


def get_mysql_info(config: configparser.ConfigParser):
    user = config.get("MYSQL", "user")
    pwd = config.get("MYSQL", "password")
    socket_path = config.get("MYSQL", "socket_path")
    return user, pwd, socket_path


def get_seed(config: configparser.ConfigParser):
    return int(config.get("RANDOM", "seed"))


def get_output_path(config: configparser.ConfigParser):
    return config.get("GENERAL", "output_path")


def get_used_databases(config: configparser.ConfigParser):
    return config.get("GENERAL", "databases").split()


def get_queries_numbers(config: configparser.ConfigParser):
    return (
        config.getint("GENERAL", "n_normal_queries"),
        config.getint("GENERAL", "n_attack_queries"),
    )


def get_statement_types_and_proportions(config: configparser.ConfigParser):
    stmts = []

    for section in config.sections():
        if section == "SQL_STATEMENTS":
            for key, value in config.items(section):
                stmts.append(
                    {"type": key, "proportion": float(Fraction(value))}
                )

    if abs(sum([stmt["proportion"] for stmt in stmts]) - 1.0) > 1e-10:
        raise ValueError(
            f"Proportions of queries types must sum up to 1. Current is {sum([stmt['proportion'] for stmt in stmts])}"
        )

    return stmts


def get_payload_types_and_proportions(config: configparser.ConfigParser):
    """Return a dictionnary of each payload type per family and its target proportion."""
    payloads = []

    for section in config.sections():
        if section == "PAYLOADS":
            for key, value in config.items(section):
                family, paytype = key.split(".")
                payloads.append(
                    {
                        "type": paytype,
                        "family": family,
                        "proportion": float(Fraction(value)),
                    }
                )
    if abs(sum([payload["proportion"] for payload in payloads]) - 1.0) > 1e-10:
        raise ValueError("Proportions of payloads types must sum up to 1.")

    return payloads
