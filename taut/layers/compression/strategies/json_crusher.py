import json
import re
from typing import Any, Dict, List, Union

from .base import CompressionResult, CompressionStrategy

def escape_val(val: Any) -> str:
    if val is None:
        return ""
    if isinstance(val, (dict, list)):
        s = json.dumps(val)
    else:
        s = str(val)
    return s.replace('\\', '\\\\').replace('|', '\\|').replace('\n', '\\n')

def replace_tables(data: Any, min_items: int) -> tuple[Any, bool]:
    changed = False
    if isinstance(data, list):
        if len(data) >= min_items and all(isinstance(x, dict) for x in data):
            first_keys = set(data[0].keys())
            if all(set(item.keys()) == first_keys for item in data):
                keys = list(data[0].keys())
                rows = []
                for item in data:
                    row_vals = [escape_val(item[k]) for k in keys]
                    rows.append(" | ".join(row_vals))
                
                header = "COLS: " + " | ".join(keys)
                table = f"{header}\n" + "\n".join(rows)
                return f"__CRUSHED_TABLE_START__\n{table}\n__CRUSHED_TABLE_END__", True
        
        new_list = []
        for x in data:
            new_x, c = replace_tables(x, min_items)
            new_list.append(new_x)
            changed = changed or c
        return new_list, changed

    elif isinstance(data, dict):
        new_dict = {}
        for k, v in data.items():
            new_v, c = replace_tables(v, min_items)
            new_dict[k] = new_v
            changed = changed or c
        return new_dict, changed
    return data, False

def unescape_table(match: re.Match) -> str:
    content = match.group(1)
    return content.replace('\\n', '\n').replace('\\"', '"')

class SmartCrusher(CompressionStrategy):
    """
    Compresses JSON arrays of objects into a columnar format recursively.
    E.g., {"users": [{"a": 1, "b": 2}, {"a": 3, "b": 4}]} -> 
    {"users": COLS: a | b
    1 | 2
    3 | 4}
    """

    def compress(self, text: str) -> CompressionResult:
        original_size = len(text)
        
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return CompressionResult(
                original_text=text,
                compressed_text=text,
                original_size=original_size,
                compressed_size=original_size,
                metadata={"error": "Invalid JSON"}
            )

        replaced_data, changed = replace_tables(data, min_items=3)
        
        if not changed:
            return CompressionResult(
                original_text=text,
                compressed_text=text,
                original_size=original_size,
                compressed_size=original_size,
                metadata={"strategy": "json_crusher", "changed": False}
            )

        # Dump without excessive spaces, but keep structure
        compressed_text = json.dumps(replaced_data, indent=2)
        
        # Unescape the table placeholders
        compressed_text = re.sub(
            r'"__CRUSHED_TABLE_START__\\n(.*?)\\n__CRUSHED_TABLE_END__"',
            unescape_table,
            compressed_text,
            flags=re.DOTALL
        )
        
        return CompressionResult(
            original_text=text,
            compressed_text=compressed_text,
            original_size=original_size,
            compressed_size=len(compressed_text),
            metadata={"strategy": "json_crusher", "changed": True}
        )
