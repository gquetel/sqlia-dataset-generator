from collections import namedtuple
import configparser
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
import re
import random
import urllib.parse

from ..sql_connector import SQLConnector
from ..config_parser import get_seed


class sqlmapGenerator:
    def __init__(
        self,
        config: configparser.ConfigParser,
        templates: pd.DataFrame,
        sqlconnector: SQLConnector,
        placeholders_dictionnaries_list: list,
        port: int,
    ):
        """Initialize data structures for payload generation."""

        self.templates = templates.to_dict("records")
        self.config = config
        self.port = port
        self.sqlc = sqlconnector
        # List of dictionnaries of values
        self.pdl = placeholders_dictionnaries_list

        self.seed = get_seed(self.config)

    def perform_attack(self, technique: tuple, template_info: dict):
        name_technique, settings_technique = technique
        default_settings = "-v 0 -D dataset --level=5 --risk=3  --skip='user-agent,referer,host' --batch --flush-session -u "
        parameters = ""
        db_name = template_info["ID"].split("-")[0]

        for i, param in enumerate(template_info["placeholders"]):
            random_param_value = random.choice(self.pdl[(db_name, param)])
            if i > 0:
                parameters += f"&{param}={random_param_value}"
            else:
                parameters += f"{param}={random_param_value}"

        url = f"\"http://localhost:{self.port}/{template_info['ID']}?{parameters.replace(' ', '%20')}\""

        # Url is built. Invoke sqlmap.
        command = "sqlmap " + settings_technique + default_settings +url
        print(command)

    def generate_attacks(self):
        techniques = {
            "boolean": "--technique=B --users ",
            "error": "--technique=E --schema --users --tables --count ",
            "union": "--technique=U -all  ",
            "stacked": "--technique=S -f ",
            "time": "--technique=T -f ",
            "inline": "--technique=Q --all ",
        }

        for template in self.templates:
            for i in techniques.items():
                self.perform_attack(i, template)
