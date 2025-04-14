from collections import defaultdict
from .payload_generators.sqlmap_generator import sqlmapGenerator
import random
from typing import Tuple, List


class PayloadDistributionManager:
    def __init__(self, payloads: list[dict], n_attack_queries: int):
        self.payloads_config = payloads
        self.total_count = 0
        self._n_attacks = n_attack_queries

        # tuple-based index dict, returning target counts for
        # given (family, type)
        self.target_counts = defaultdict(int)
        # tuple-based index dict, returning current counts for
        # given (family, type)
        self.current_counts = defaultdict(int)

        self._family_types = set()
        for payload in self.payloads_config:
            self.target_counts[(payload["family"], payload["type"])] = (
                payload["proportion"] * self._n_attacks
            )
            self.current_counts[(payload["family"], payload["type"])] = 0
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
                self.generators[family] = sqlmapGenerator
            else:
                raise ValueError(f"No configuration for family '{family}'")

    def _get_current_proportion(self, family, payload_type):
        """Get the current proportion of a specific type across all payloads."""
        if self.total_count == 0:
            return 0.0
        return self.current_counts[(family, payload_type)] / self.total_count

    def select_next_family_and_type(self):
        if self.total_count == 0:
            print(self.current_counts.keys())
        return "", ""

    def generate_payload(self):
        family, payload_type = self.select_next_family_and_type()
