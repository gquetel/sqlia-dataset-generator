# Abstract class for payload generation

from abc import ABC, abstractmethod

class PayloadGenerator(ABC):
    @abstractmethod
    def generate_payload_from_type(self, type: str) -> tuple[str, str]:
        pass
