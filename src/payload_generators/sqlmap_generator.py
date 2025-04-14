from .payload_generator import PayloadGenerator

class sqlmapGenerator(PayloadGenerator):
    def __init__(self):
        """Initialize data structures for payload generation."""
        self._tmp = None

    def generate_payload_from_type(self, type):
        """ Return malicious payload of given type. """
        payload = ""
        desc = ""
        return (payload, desc)
