# Abstract class for payload generation

from abc import ABC, abstractmethod
from collections import namedtuple


class PayloadGenerator(ABC):
    @abstractmethod
    def generate_payload_from_type(
        self, original_value: str | int, payload_type: str, payload_clause: namedtuple
    ) -> tuple[str, str]:
        pass
    
    @abstractmethod
    def get_possible_types_from_clause(self, clause : str) -> list: 
        pass