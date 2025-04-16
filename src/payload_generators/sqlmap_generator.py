import os
import pandas as pd
import xml.etree.ElementTree as ET

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
    def __init__(self):
        """Initialize data structures for payload generation."""
        # First, we load all xml files
        self.payloads = {}
        self._load_all_payloads()

    def _load_all_payloads(self):
        # Read all files under ./data/payloads/sqlmap ending with .xml
        payloads_dir = "./data/payloads/sqlmap/"
        accepted_clauses = set([0, 1])
        for filename in os.listdir(payloads_dir):
            if filename.endswith(".xml"):
                payload_family = []
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
                        # - <request>, the payload to use, and some of its settings.
                        # <details/dbms>, to check wether the payload can by used by MySQL servers.

                        # We only select MySQL applicable payloads.
                        # If no details/dbms => Is generic
                        # Else, evaluate wether MySQL is present in field.
                        dbms_node = child.find("details/dbms")
                        if dbms_node is None or "MySQL" in dbms_node.text:
                            title = child.find("title").text
                            clauses = set(_normalize_ranges(child.find("clause").text))
                            if(9 in clauses):
                                print(title)
                            where = child.find("where").text
                            request_payload = child.find("request/payload").text

                            clauses = clauses.intersection(accepted_clauses)
                            if len(clauses) > 0:
                                payload_family.append(
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
                self.payloads[filename[:-4]] = pd.DataFrame(payload_family)

    def _map_clause_name_to_int(clause_name : str )-> int:
        match(clause_name):
            case "where":
                pass
            case "values":
                pass
    def generate_payload_from_type(
        self, payload_type: str, payload_clause
    ) -> tuple[str, str]:
        """Return malicious payload of given type."""
        payload = "sqlmap_payload"
        _id = "sqlmap_id"
        return (payload, _id)
