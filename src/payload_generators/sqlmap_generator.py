from .payload_generator import PayloadGenerator


class sqlmapGenerator(PayloadGenerator):
    def __init__(self):
        """Initialize data structures for payload generation."""
        self._tmp = None

    def generate_payload_from_type(self, type) -> tuple[str, str]:
        """Return malicious payload of given type."""
        payload = "sqlmap_payload"
        _id = "sqlmap_id"
        return (payload, _id)
