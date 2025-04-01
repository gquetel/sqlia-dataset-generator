import os
import random

FILEPATHS = {
    "datadir:": "./data/",
    "databasesdir": "./data/databases/",
    "payloadsdir": "./data/payloads/",
}

def pick_from_dict(d_dictionnaries: dict, type: str) -> str:
    return random.choice(d_dictionnaries[type])


def construct_normal_query(l_templates: list, d_dictionnaries: dict) -> str:
    picked_template = random.choice(l_templates)
    for dictionnary in d_dictionnaries:
        picked_template = picked_template.replace(
            f"{{{dictionnary}}}", pick_from_dict(d_dictionnaries, dictionnary)
        )
        picked_template = picked_template.replace(
            f"{{!{dictionnary}}}", pick_from_dict(d_dictionnaries, dictionnary)
        )
    return picked_template


def load_dictionnaries(database: str) -> dict:
    dictionnaries = {}
    fp_dictionnaries = FILEPATHS["databasesdir"] + database + "/dictionnaries/"

    # Iterate over all files in the dictionnaries folder
    for filename in os.listdir(fp_dictionnaries):
        with open(fp_dictionnaries + filename, "r") as f:
            dictionnaries[filename] = f.read().splitlines()

    return dictionnaries


def load_payloads(payloads_to_use: list) -> list:
    payloads = []

    for payload in payloads_to_use:
        payload_fp = (
            FILEPATHS["payloadsdir"] + payload["family"] + "/" + payload["type"]
        )

        if not os.path.exists(payload_fp):
            raise ValueError(
                f"Specified payload file '{payload['family']}/{payload['type']}' has not been found in '{FILEPATHS['payloadsdir']}'."
            )

        with open(payload_fp, "r") as f:
            payloads.append(
                {
                    "type": payload["type"],
                    "family": payload["family"],
                    "proportion": payload["proportion"],
                    "templates": f.read().splitlines(),
                }
            )
    return payloads


def load_query_templates(database: str, query_type: str) -> list:
    fp_templates = FILEPATHS["databasesdir"] + database + "/queries/" + query_type
    with open(fp_templates, "r") as f:
        return f.read().splitlines()
