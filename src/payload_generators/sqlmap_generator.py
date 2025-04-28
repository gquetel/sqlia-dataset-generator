import configparser
import os
import pandas as pd
import re
import string
import random
import xml.etree.ElementTree as ET

from ..config_parser import get_payload_types_and_proportions, get_seed

from .payload_generator import PayloadGenerator


def _normalize_ranges(text: str):
    nums = []
    for item in text.split(","):
        pre = item.split("-")
        if len(pre) == 1:
            nums.append(int(pre[0]))
        else:
            nums.extend(list(range(int(pre[0]), int(pre[1]) + 1)))
    return nums


class sqlmapGenerator(PayloadGenerator):
    def __init__(self, config: configparser.ConfigParser):
        """Initialize data structures for payload generation."""
        # First, we load all payload xml files

        payload_config = get_payload_types_and_proportions(config)
        _valid_payloads = [d["type"] for d in payload_config if d["family"] == "sqlmap"]
        self.payloads = {}
        self._load_all_payloads(_valid_payloads)
        self.config = config
        self.seed = get_seed(self.config)

    def get_possible_types_from_clause(self, clause: str) -> list:
        all_types = [
            "boolean_blind",
            "error_based",
            "inline_query",
            "stacked_queries",
            "time_blind",
            # "union_query", # TODO, support union.
        ]
        match (clause):
            case "where":
                return all_types
            case "values":
                # TODO: voir si on peut rajouter d'autres injections qui font sens.
                return ["stacked_queries"]
            case _:
                raise ValueError(
                    "get_possible_types_from_clause: unknown clause ", clause
                )

    def _load_all_payloads(self, valid_payloads: list):
        # Read all files under ./data/payloads/sqlmap ending with .xml
        payloads_dir = "./data/payloads/sqlmap/"
        accepted_clauses = set([0, 1])
        for filename in os.listdir(payloads_dir):
            if filename.endswith(".xml") and filename[:-4] in valid_payloads:
                payload_type = []
                tree = ET.parse(payloads_dir + filename)
                root = tree.getroot()
                # root = data, a list of tests.
                for child in root:
                    try:
                        # child = test, with payload properties as sub-tags
                        # We need to save informations about:
                        # - <title>, the description of the payload.
                        # - <clause>, we only collect clause for WHERE or Always. Subject to evolution
                        # - <where>, what to do with parameter original value
                        # - request/payload, the payload to use, and some of its settings.
                        # - request/comment, if exist, suffix comment char to payload.
                        # <details/dbms>, to check wether the payload can by used by MySQL servers.

                        # We only select MySQL applicable payloads.
                        # If no details/dbms => Is generic
                        # Else, evaluate wether MySQL is present in field.

                        dbms_node = child.find("details/dbms")
                        if dbms_node is None or "MySQL" in dbms_node.text:
                            title = child.find("title").text
                            clauses = set(_normalize_ranges(child.find("clause").text))
                            where = child.find("where").text
                            request_payload = child.find("request/payload").text

                            comment = child.find("request/comment")
                            if comment:
                                request_payload += comment.text

                            clauses = clauses.intersection(accepted_clauses)
                            if len(clauses) > 0:
                                payload_type.append(
                                    {
                                        "title": title,
                                        "clauses": clauses,
                                        "where": int(where),
                                        "payload": request_payload,
                                    }
                                )

                    except AttributeError:
                        title = (
                            child.find("title").text
                            if child.find("title") is not None
                            else child.tag
                        )
                        print(
                            "_load_all_payloads: failed to fetch all informations for child: ",
                            title,
                        )
                print(
                    f"Found {len(payload_type)} payloads templates of type {filename[:-4]}"
                )
                self.payloads[filename[:-4]] = pd.DataFrame(payload_type)

    def _fill_payload_template(self, original_value: str | int, template: str) -> str:
        sqlmap_fields = re.findall(r"(\[.*?\])", template)
        for field in sqlmap_fields:
            match field:
                case (
                    "[RANDNUM]"
                    | "[RANDNUM1]"
                    | "[RANDNUM2]"
                    | "[RANDNUM3]"
                    | "[RANDNUM4]"
                    | "[RANDNUM5]"
                ):
                    template = template.replace(field, str(random.randint(0, 10000)))
                case "[SLEEPTIME]":
                    template = template.replace(field, str(random.randint(0, 5)), 1)
                case "[ORIGVALUE]":
                    if isinstance(original_value, str):
                        escape_char = random.choice(['"', "'"])
                        template = template.replace(
                            field,
                            f"{escape_char}{original_value}{escape_char}",
                        )
                    elif isinstance(original_value, int) or isinstance(
                        original_value, float
                    ):
                        template = template.replace(field, str(original_value))
                    else:
                        raise ValueError(
                            "_fill_payload_template: Unknown type for original_value ",
                            type(original_value),
                        )
                case "[RANDSTR]":
                    _rdmstr_len = random.randint(5, 30)
                    _rdmstr = "".join(
                        random.choices(
                            string.digits + string.ascii_letters, k=_rdmstr_len
                        )
                    )
                    template = template.replace(field, _rdmstr, 1)
                case "[DELIMITER_START]" | "[DELIMITER_STOP]":
                    # In SQLMAP corresponds to 3 low frequency characters that act as delimiters, to be easily recognized in the response.
                    # Here we don't mind about the responds, let's just replace it with random characters.
                    _rdmstr = "".join(
                        random.choices(string.digits + string.ascii_letters, k=3)
                    )
                    template = template.replace(field, _rdmstr, 1)
                case _:
                    print("_fill_payload_template: Unknown field:", field)
        return template

    def generate_payload_from_type(
        self, original_value: str | int, payload_type: str, payload_clause: str
    ) -> tuple[str, str]:
        payload = None
        desc = None
        where = None

        escape_char = random.choice(['"', "'"])
        comments_char = random.choice(["#", "--"])

        # Randomly select a payload in self.payloads[payload_type]
        # When a string is expected, we avoid type 2 where payloads.
        # Also, escape escape_char in original_value.
        if isinstance(original_value, str):
            original_value = original_value.replace(
                escape_char, f"{escape_char}{escape_char}"
            )
            _df = self.payloads[payload_type]
            _df = _df[_df["where"] != 2]
            assert len(_df) > 0  # No such case has been seen so far
            _choosen_payload = _df.sample(n=1)
        else:
            _choosen_payload = self.payloads[payload_type].sample(n=1)

        desc = _choosen_payload.iloc[0]["title"]
        where = _choosen_payload.iloc[0]["where"]

        payload = self._fill_payload_template(
            original_value=original_value,
            template=_choosen_payload.iloc[0]["payload"],
        )
        # Now we need to build prefix / suffix depending on value type and clause
        # if clause = values, we only support stacked queries, we need to close parenthesis.
        if payload_clause == "values":
            if isinstance(original_value, str):
                # Escape quoting char in original value
                payload = escape_char + original_value + escape_char + ")" + payload
            elif isinstance(original_value, int) or isinstance(original_value, float):
                payload = str(original_value) + ")" + payload
            else:
                raise ValueError(
                    "generate_payload_from_type: original_value is of unknown type:",
                    type(original_value),
                )

            # Then add comments to ignore rest of query.
            # TODO, append random comments.
            payload += comments_char
        else:
            if isinstance(original_value, str):
                match (where):
                    case 1:
                        #  Append the payload to the parameter original value
                        payload = (
                            escape_char + original_value + escape_char + " " + payload
                        )
                    case 3:
                        # Replace the parameter original value with payload
                        # No need to escape ?
                        pass
                    case _:
                        raise ValueError(
                            "generate_payload_from_type: where is of incorrect value:"
                        )
            elif isinstance(original_value, int) or isinstance(original_value, float):
                match (where):
                    case 1:
                        #  Append the payload to the parameter original value
                        payload = str(original_value) + " " + payload
                    case 2:
                        # Random negative int + payload
                        payload = str(random.randint(-100000, -1000)) + " " + payload
                    case 3:
                        # Replace the parameter original value with our payload
                        pass
                    case _:
                        raise ValueError(
                            "generate_payload_from_type: where is of incorrect value:"
                        )
            else:
                raise ValueError(
                    "generate_payload_from_type: original_value is of unknown type:",
                    type(original_value),
                )
        return (payload, desc)

    def generate_undefined_from_type(
        self,
        original_value: str | int,
        payload_type: str,
        payload_clause: str,
    ) -> tuple[str, str]:
        """Return malicious payload of given type."""
        # TODO: remove scenario from args.

        payload = None
        desc = None
        where = None

        escape_char = random.choice(['"', "'"])
        comments_char = random.choice(["# ", "-- "])

        # TODO, add weights for higher occurence of improper escaping.

        match (payload_type):
            case "boolean_blind" | "error_based" | "inline_query" | "time_blind":
                scenario = random.choices(
                    population=["escaping", "comment", "keyword", "truncate"],
                    weights=[10, 3, 1, 1],
                )[0]
            case "stacked_queries":
                scenario = random.choices(
                    population=["comment", "keyword", "truncate", "no_semicolon"],
                    weights=[3, 1, 1, 3],
                )[0]

        # Randomly select a payload in self.payloads[payload_type]
        # When a string is expected, we avoid type 2 where payloads.
        # Also, escape escape_char in original_value.
        if isinstance(original_value, str):
            original_value = original_value.replace(
                escape_char, f"{escape_char}{escape_char}"
            )
            _df = self.payloads[payload_type]
            _df = _df[_df["where"] != 2]
            assert len(_df) > 0  # No such case has been seen so far
            _choosen_payload = _df.sample(n=1)
        else:
            _choosen_payload = self.payloads[payload_type].sample(n=1)

        where = _choosen_payload.iloc[0]["where"]

        payload = self._fill_payload_template(
            original_value=original_value,
            template=_choosen_payload.iloc[0]["payload"],
        )

        if scenario == "no_semicolon":
            payload = payload.replace(";", " ")
            desc = "Improper stacked queries construction"

        if scenario == "comment":
            _w = payload.split()
            _random_i = random.randint(0, len(_w) - 1)
            _w[_random_i] = random.choice(["# ", "-- "]) + _w[_random_i]
            payload = " ".join(_w)
            desc = "Improper comment addition"

        if scenario == "keyword":
            keywords = ["OR ", "AND ", "HAVING ", "WHERE ", "UNION"]
            _w = payload.split()
            _random_i = random.randint(0, len(_w) - 1)
            _w[_random_i] = random.choice(keywords) + _w[_random_i]
            payload = " ".join(_w)
            desc = "Presence of invalid SQL keyword"

        if scenario == "truncate":
            _w = payload.split()
            # -2 so that we never select the full query.
            try:
                _random_i = random.randint(0, len(_w) - 2)
                payload = " ".join(_w[:_random_i])
            except ValueError as e:
                # No space, cut at random
                _random_i = random.randint(0, len(payload) - 1)
                payload = payload[:_random_i]
            desc = "Incorrect truncation of query"

        if payload_clause == "values":
            if isinstance(original_value, str):
                # Escape quoting char in original value
                payload = escape_char + original_value + escape_char + ")" + payload
            elif isinstance(original_value, int) or isinstance(original_value, float):
                payload = str(original_value) + ")" + payload
            else:
                raise ValueError(
                    "generate_undefined_from_type: original_value is of unknown type:",
                    type(original_value),
                )

            # Then add comments to ignore rest of query.
            # TODO, append random comments.
            payload += comments_char
        else:
            if isinstance(original_value, str):
                match (where):
                    case 1:
                        #  Append the payload to the parameter original value
                        if scenario == "escaping":
                            desc = "Improper escaping strategy"
                            _inv_esc_char_list = [
                                '")',
                                '"))',
                                "')",
                                "'))",
                                "''",
                                "`",
                                "'\"",
                                "`'",
                            ]

                            if escape_char == "'":
                                invalid_escape_char = random.choice(
                                    _inv_esc_char_list + ['"']
                                )
                            elif escape_char == '"':
                                invalid_escape_char = random.choice(
                                    _inv_esc_char_list + ["'"]
                                )
                            payload = (
                                escape_char
                                + original_value
                                + invalid_escape_char
                                + " "
                                + payload
                            )

                        else:
                            payload = (
                                escape_char
                                + original_value
                                + escape_char
                                + " "
                                + payload
                            )
                    case 3:
                        # Replace the parameter original value with payload
                        # No need to escape ?
                        pass
                    case _:
                        raise ValueError(
                            "generate_undefined_from_type: where is of incorrect value:"
                        )
            elif isinstance(original_value, int) or isinstance(original_value, float):
                match (where):
                    case 1:
                        #  Append the payload to the parameter original value
                        if scenario == "escaping":
                            # Add invalid string quote
                            desc = "Improper escaping strategy"
                            payload = str(original_value) + "'" + payload

                        else:
                            payload = str(original_value) + " " + payload

                    case 2:
                        # Random negative int + payload
                        if scenario == "escaping":
                            # Add invalid string quote
                            desc = "Improper escaping strategy"
                            payload = (
                                str(random.randint(-100000, -1000)) + '"' + payload
                            )
                        else:
                            payload = (
                                str(random.randint(-100000, -1000)) + " " + payload
                            )
                    case 3:
                        # Replace the parameter original value with our payload
                        if scenario == "escaping":
                            # Add invalid string quote
                            desc = "Improper escaping strategy"
                            payload += "'"
                        else:
                            pass
                    case _:
                        raise ValueError(
                            "generate_undefined_from_type: where is of incorrect value:"
                        )
            else:
                raise ValueError(
                    "generate_undefined_from_type: original_value is of unknown type:",
                    type(original_value),
                )

        return (payload, desc)
