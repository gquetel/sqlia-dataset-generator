import configparser
import pandas as pd
import os
import string
import random

from ..config_parser import get_payload_types_and_proportions, get_seed

from .payload_generator import PayloadGenerator


class crawledGenerator(PayloadGenerator):
    def __init__(self, config: configparser.ConfigParser):
        self.config = config
        self.seed = get_seed(self.config)

        payload_config = get_payload_types_and_proportions(self.config)
        _valid_payloads = [
            d["type"] for d in payload_config if d["family"] == "crawled"
        ]
        self.payloads = {}
        self._load_all_payloads(_valid_payloads)

    def _load_all_payloads(self, valid_payloads: list):
        payloads_dir = "./data/payloads/crawled/"
        for filename in os.listdir(payloads_dir):
            if filename.endswith(".csv") and filename[:-4] in valid_payloads:
                df_payload_type = pd.read_csv(payloads_dir + filename)
                self.payloads[filename[:-4]] = df_payload_type

    def generate_payload_from_type(
        self, original_value: str | int, payload_type: str, payload_clause: str
    ) -> tuple[str, str]:
        payload = None
        desc = None

        escape_char = random.choice(['"', "'"])
        comments_char = random.choice(["# ", "-- "])
        # Randomly select one payload of type 'payload_type' amongst our collection.
        _choosen_payload = self.payloads[payload_type].sample(n=1)
        desc = _choosen_payload.iloc[0]["desc"]
        payload = _choosen_payload.iloc[0]["payload"]

        # All of the remaining code expect that payload type is "inband"
        # Once we have other types I should create functions to route
        # the payload generation depending on the asked type.

        # If we are in an INSERT statement, we'll try to close the parenthesis
        # This only work if we are in last field, else syntax error.
        if payload_clause == "values":
            if isinstance(original_value, str):
                payload = (
                    escape_char + original_value + escape_char + ") ;" + payload
                )
            elif isinstance(original_value, int) or isinstance(
                original_value, float
            ):
                payload = str(original_value) + ")" + payload
            else:
                raise ValueError(
                    "generate_payload_from_type: original_value is of unknown type:",
                    type(original_value),
                )

        else:
            # Otherwise we evalue type of original_value to escape correctly
            if isinstance(original_value, str):
                payload = (
                    escape_char + original_value + escape_char + " ;" + payload
                )
            elif isinstance(original_value, int) or isinstance(
                original_value, float
            ):
                payload = str(original_value) + "; " + payload
            else:
                raise ValueError(
                    "generate_payload_from_type: original_value is of unknown type:",
                    type(original_value),
                )
            
        # Inband query: we need to comment all that is coming after the
        # injected inband statement;
        payload += comments_char
        return (payload, desc)

    def get_possible_types_from_clause(self, clause: str) -> list:
        all_types = ["inband"]
        return all_types
