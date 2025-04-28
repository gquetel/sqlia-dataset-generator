# Abstract class for payload generation

from abc import ABC, abstractmethod


class PayloadGenerator(ABC):
    @abstractmethod
    def generate_payload_from_type(
        self, original_value: str | int, payload_type: str, payload_clause: str
    ) -> tuple[str, str]:
        pass

    @abstractmethod
    def generate_undefined_from_type(
        self,
        original_value: str | int,
        payload_type: str,
        payload_clause: str,
    ) -> tuple[str, str]:
        pass
