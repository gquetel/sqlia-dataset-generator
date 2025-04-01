from .payload_generator import PayloadGenerator
from ..queries_generator import (
    construct_normal_query,
    load_query_templates,
    load_dictionnaries,
)
import re
import random
import string


def generate_random_string(length: int = -1) -> str:
    if length == -1:
        length = random.randint(5, 20)
    return "".join(
        random.choices(string.digits + string.ascii_letters, k=length)
    )


class sqlmapGenerator(PayloadGenerator):
    def __init__(
        self, payload_templates: list, d_dictionnaries, database
    ) -> None:
        self.payload_templates = payload_templates
        self.d_dictionnaries = d_dictionnaries
        self.database = database

        self.escape_sequences = ["'", '"']
        self.comment_sequences = ["-- ", "#"]
        self.padding_space = ["", " ", "\t"]
        self.inference_symbols = ["=", "!=", ">", ">=", "<", "<="]

    def generate_payload_prefix(self) -> str:
        """Adds escape sequence for payload to be interpreted.

        Returns:
            str: _description_
        """
        return random.choice(self.escape_sequences)

    # TODO: Is useless ?
    def generate_payload_suffix(self) -> str:
        """Return a random comment to close the payload.

        Returns:
            str: _description_
        """
        return (
            ";"
            + random.choice(self.padding_space)
            + random.choice(self.comment_sequences)
        )

    def replace_placeholder(
        self, payload_template: str, placeholder: str, replacement: str
    ) -> str:
        return payload_template.replace(placeholder, replacement, 1)

    def deal_with_where(
        self, where_field: str, payload_template: str, placeholder
    ) -> str:
        payload_template = payload_template.replace(where_field, "")
        behavior = int(
            where_field[-2]
        )  # placeholder is of the form {WHEREX} X in [1, 2, 3]

        match behavior:
            case 1:
                # Append the string to the parameter original value.
                legitimate_part = random.choice(
                    self.d_dictionnaries[placeholder[1:-1]]
                )
                payload_template = legitimate_part + " " + payload_template
            case 2:
                # Repalce the original value with a negative random integer value and append our string.
                payload_template = (
                    str(random.randint(-10000, 0)) + " " + payload_template
                )
            case 3:
                # Replace the parameter original value with our string.
                pass
            case _:
                raise ValueError("Invalid behavior value")
        return payload_template

    def deal_with_query(self, payload_template, field, placeholder):
        # Replace {QUERY} with a MySQL specific error query.

        normal_query = construct_normal_query(l_templates, d_dict)

    def fill_payload_template(
        self, payload_template: str, placeholder: str
    ) -> str:
        # Field = SQLMAP placeholder to replace
        # placeholder = Our skeleton query placeholder to be filled with a dict value.

        sqlmap_fields = re.findall(r"({.*?})", payload_template)
        for field in sqlmap_fields:
            match field:
                case "{RANDNUM}":
                    payload_template = self.replace_placeholder(
                        payload_template, field, str(random.randint(0, 10000))
                    )

                case "{QUERY}":
                    # TODO
                    pass
                case "{SLEEPTIME}":
                    payload_template = self.replace_placeholder(
                        payload_template, field, str(random.randint(1, 10))
                    )
                case "{RANDSTR}":
                    payload_template = self.replace_placeholder(
                        payload_template, field, generate_random_string()
                    )
                case "{DELIMITER_START}" | "{DELIMITER_STOP}":
                    # In SQLMAP corresponds to 3 low frequency characters that act as delimiters, to be easily recognized in the response.
                    # Here we don't mind about the responds, let's just replace it with random characters.
                    payload_template = self.replace_placeholder(
                        payload_template,
                        field,
                        generate_random_string(length=3),
                    )

                case "{WHERE1}" | "{WHERE2}" | "{WHERE3}":
                    payload_template = self.deal_with_where(
                        field, payload_template, placeholder
                    )

                case "{INFERENCE}":
                    inference_string = (
                        str(random.randint(0, 10000))
                        + random.choice(self.inference_symbols)
                        + str(random.randint(0, 10000))
                    )
                    payload_template = self.replace_placeholder(
                        payload_template, field, inference_string
                    )
                case _:
                    pass

        remaining_placeholders = re.findall(r"{.*?}", payload_template)
        print(remaining_placeholders)
        return payload_template

    def build_payload(self, placeholder: str) -> str:
        payload_template = random.choice(self.payload_templates)

        # Payload behavior is depicted in:
        # https://github.com/sqlmapproject/sqlmap/blob/master/data/xml/payloads/boolean_blind.xml

        full_payload = (
            self.generate_payload_prefix()
            + self.fill_payload_template(payload_template, placeholder)
        )
        return full_payload

    def inject_payload_in_query_template(self, query_template: str) -> str:
        placeholders = [
            (m.start(), m.end(), m.group(0))
            for m in re.finditer(r"{[^!].*?}", query_template)
        ]
        # Randomly select one injection site
        start, end, placeholder = random.choice(placeholders)
        payload = self.build_payload(placeholder)

        query_template = (
            query_template[:start] + payload + query_template[end:]
        )  # end - 1 to remove the closing bracket

        return query_template

    def generate_payload(self, query_template: str) -> str:
        # From a query template, identify placeholders to inject a payload.
        return self.inject_payload_in_query_template(query_template)
