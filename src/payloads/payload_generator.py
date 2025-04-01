# Abstract class for payload generation 

from abc import ABC, abstractmethod


class PayloadGenerator(ABC):
    @abstractmethod
    def generate_payload(self, query_template: str, database) -> str:
        pass
    