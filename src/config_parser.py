import configparser
from fractions import Fraction


def get_mysql_info(config: configparser.ConfigParser):
    user = config.get("MYSQL", "user")
    pwd = config.get("MYSQL", "password")
    socket_path = config.get("MYSQL", "socket_path")
    root_password = config.get("MYSQL","root_password")
    return user, pwd, socket_path,root_password

def get_seed(config: configparser.ConfigParser):
    return int(config.get("GENERAL", "seed"))

def get_attacks_ratio(config: configparser.ConfigParser):
    return float(Fraction(config.get("GENERAL", "attacks_ratio")))


def get_output_path(config: configparser.ConfigParser):
    return config.get("GENERAL", "output_path")


def get_used_databases(config: configparser.ConfigParser):
    return config.get("GENERAL", "databases").split()


def get_statement_types_and_proportions(config: configparser.ConfigParser):
    stmts = []

    for section in config.sections():
        if section == "NORMAL_TRAFFIC_TARGETS":
            for key, value in config.items(section):
                stmts.append({"type": key, "proportion": float(Fraction(value))})

    if abs(sum([stmt["proportion"] for stmt in stmts]) - 1.0) > 1e-10:
        raise ValueError(
            f"Proportions of queries types must sum up to 1. Current is {sum([stmt['proportion'] for stmt in stmts])}"
        )

    return stmts
