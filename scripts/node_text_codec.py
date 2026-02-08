import json

SPACE_REPL = "_"

def encode_node(s: str) -> str:
    s = str(s).lower().replace("\n", " ").replace("\t", " ").strip()
    while "  " in s:
        s = s.replace("  ", " ")
    return s.replace(" ", SPACE_REPL)

def decode_node(s: str) -> str:
    return s.replace(SPACE_REPL, " ")

def save_mapping(mapping, out_json):
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
