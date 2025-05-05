from collections import namedtuple
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
            "union_query",
        ]
        return all_types

    def _sanitize_payload(self, payload: str):
        # For some payloads, there is a '\' that needs to be replaced to
        # a '\\' so that it does not become syntactically invalid.
        payload = payload.replace("'\\'", "'\\\\'")
        return payload

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
                            request_payload = self._sanitize_payload(request_payload)

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
                # print(
                #     f"Found {len(payload_type)} payloads templates of type {filename[:-4]}"
                # )
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
                    # template = template.replace(field, str(random.randint(0, )), 1)
                    template = template.replace(field, "0")
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
        self, original_value: str | int, payload_type: str, query_template: namedtuple
    ) -> tuple[str, str]:
        payload = None
        desc = None
        where = None

        payload_clause = query_template.payload_clause

        # Try to retrieve information provided by select statement
        # When not present -> Other type of statement, will lead to
        # Syntax error anyway, we chose a random value.
        if hasattr(query_template, "expected_column_number"):
            payload_expected_cols = query_template.expected_column_number
        else:
            payload_expected_cols = random.randint(1, 11)

        escape_char = random.choice(['"', "'"])
        comments_char = random.choice(["#", "--"])

        # Randomly select a payload in self.payloads[payload_type]
        # When a string is expected, we avoid type 2 where payloads.
        # Also, escape escape_char in original_value.
        if isinstance(original_value, str) and payload_type != "union_query":
            original_value = original_value.replace(
                escape_char, f"{escape_char}{escape_char}"
            )
            _df = self.payloads[payload_type]
            _df = _df[_df["where"] != 2]
            assert len(_df) > 0  # No such case has been seen so far
            _choosen_payload = _df.sample(n=1)
        else:
            _choosen_payload = self.payloads[payload_type].sample(n=1)



        # Hack, Union queries templates are different
        # Just construct them directly based on payload_expected_cols.
        if payload_type == "union_query":
            choice = random.choice([random.randint(1, 10000), "NULL"])
            desc = "UNION ALL based injection attack."
            payload = (
                "UNION ALL SELECT "
                + f"{choice}," * (payload_expected_cols - 1)
                + f"{choice} -- "
            )
        else:
            desc = _choosen_payload.iloc[0]["title"]
            where = _choosen_payload.iloc[0]["where"]
            
            payload = self._fill_payload_template(
                original_value=original_value,
                template=_choosen_payload.iloc[0]["payload"],
            )

        # if(random.choice([0,1]) == 1):
        #     payload = str.lower(payload)

        # Now we need to build prefix / suffix depending on value type and clause.  .
        if payload_clause == "values":
            # we need to try to close parenthesis of INSERT ... INTO (...)
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
            payload += comments_char
        elif payload_clause == "where" or payload_clause == "subquery where":
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
        else:
            raise ValueError(
                "generate_payload_from_type: payload_clause is of unknown type:",
                payload_clause,
            )
        return (payload, desc)
