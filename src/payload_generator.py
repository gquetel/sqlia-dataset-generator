from collections import defaultdict
import configparser

from .config_parser import get_payload_types_and_proportions, get_queries_numbers
from .payload_generators.sqlmap_generator import sqlmapGenerator
import random
from typing import Tuple, List

# TODO: Refaire une passe sur les variables nécéssaires
# ATM, target_counts & total_count ne sont pas utilisés.


class PayloadDistributionManager:
    def __init__(
        self,
        config: configparser.ConfigParser,
    ):
        self.payloads_config = get_payload_types_and_proportions(config)
        self.total_count = 0
        _, n_a, _ = get_queries_numbers(config=config)
        self._n_attacks = n_a
        self.config = config
        # tuple-based index dict, returning target counts for
        # given (family, type)
        self.target_counts = defaultdict(int)
        # tuple-based index dict, returning counter of remaining payloads
        # to be generated given (family, type)
        self.remaining_counts = defaultdict(int)

        self._family_types = set()
        for payload in self.payloads_config:
            target = int(payload["proportion"] * self._n_attacks)
            self.target_counts[(payload["family"], payload["type"])] = target
            self.remaining_counts[(payload["family"], payload["type"])] = target
            self._family_types.add(payload["family"])

        self.generators = {}
        self._cache_next_types = List[Tuple[str, str]]
        self._init_generators()

    def _init_generators(self):
        """Init all necessary payload generators for dataset generation.

        Raises:
            ValueError: _description_
        """
        for family in self._family_types:
            if family == "sqlmap":
                self.generators[family] = sqlmapGenerator(config=self.config)
            else:
                raise ValueError(f"No configuration for family '{family}'")

    def _select_next_family_and_type(self, clause: str):
        # Randomly select a family and type using remaining_count as weights

        # First, fetch all injectable payloads for each family for
        # current clause.
        possible_indexes = []
        for family in self._family_types:
            possible_payloads = self.generators[family].get_possible_types_from_clause(
                clause
            )

            # Only study valid keys
            possible_indexes.extend([(family, type) for type in possible_payloads])

        valid_keys = [
            key for key in self.remaining_counts.keys() if key in possible_indexes
        ]
        assert len(valid_keys) > 0

        # Get weights for valid keys only
        weights = [self.remaining_counts[key] for key in valid_keys]

        if all(weight <= 0 for weight in weights):
            # randomly select one, it is better to deviate from
            # distribution than not generating many payloads.
            choice = random.choices(valid_keys, k=1)[0]
        else:
            # replace neg value by 0 to prevent sum(weight) == 0
            weights = [max(0, w) for w in weights]
            choice = random.choices(valid_keys, weights=weights, k=1)[0]

        self.total_count += 1
        self.remaining_counts[choice] -= 1
        return choice

    def get_final_stats(self) -> str:
        s = f"Generated {self.total_count} attacks."
        for index in self.target_counts.keys():
            f, t = index
            generated = self.target_counts[index] - self.remaining_counts[index]
            s += "\n" + f"- {generated} {t} attacks from {f}."
        return s

    def generate_payload(
        self, original_value: str | int, clause: str
    ) -> tuple[str, str]:
        # Given the clause, randomly select a possible payload
        # If possible payload count is target count, still generate.

        family, payload_type = self._select_next_family_and_type(clause)

        return self.generators[family].generate_payload_from_type(
            original_value, payload_type, clause
        )
