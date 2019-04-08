import json


def load_json(file):
    with open(file, "r") as f:
        return json.load(f)