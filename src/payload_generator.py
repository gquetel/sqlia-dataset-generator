from collections import defaultdict
from .payload_generators.sqlmap_generator import sqlmapGenerator
import random
from typing import Tuple, List

# TODO: Refaire une passe sur les variables nécéssaires
# ATM, target_counts & total_count ne sont pas utilisés.


class PayloadDistributionManager:
    def __init__(self, payloads: list[dict], n_attack_queries: int):
        self.payloads_config = payloads
        self.total_count = 0
        self._n_attacks = n_attack_queries

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
                self.generators[family] = sqlmapGenerator()
            else:
                raise ValueError(f"No configuration for family '{family}'")

    def select_next_family_and_type(self):
        # Randomly select a family and type using remaining_count as weights
        keys = list(self.remaining_counts.keys())
        weights = self.remaining_counts.values()
        choice = random.choices(keys, weights=weights, k=1)[0]

        # Increment total, update weights.
        self.total_count += 1
        self.remaining_counts[choice] -= 1
        return choice

    def generate_payload(self, clause: str) -> tuple[str, str]:
        family, payload_type = self.select_next_family_and_type()

        return self.generators[family].generate_payload_from_type(payload_type, clause)
