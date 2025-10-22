import json
import os
import pickle
from typing import Any, Generator


def load_json(file_path: str, default: list | dict | None = None) -> list | dict:
    """Load a JSON file.

    Args:
        file_path: The file path.
        default: The default value if the file does not exist.

    Returns:
        The loaded JSON data.
    """
    if default is not None and not os.path.exists(file_path):
        return default
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def save_json(data: list | dict, file_path: str):
    """Save a JSON file.

    Args:
        data: The data to save.
        file_path: The file path.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def load_jsonl(file_path: str, default: list | None = None) -> list[dict]:
    """Load a JSONL file.

    Args:
        file_path: The file path.
        default: The default value if the file does not exist.

    Returns:
        The loaded JSONL data.
    """
    if default is not None and not os.path.exists(file_path):
        return default
    with open(file_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    return data


def read_jsonl(file_path: str, default: list[dict] | None = None) -> Generator[dict, None, None]:
    """Read a JSONL file line by line.

    Args:
        file_path: The file path.
        default: The default value if the file does not exist.

    Returns:
        The generator of loaded JSONL data.
    """
    if default is not None and not os.path.exists(file_path):
        yield from default
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)  # type: ignore


def save_jsonl(data: list[dict], file_path: str):
    """Save a JSONL file.

    Args:
        data: The data to save.
        file_path: The file path.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_obj(file_path: str, default: Any = None) -> Any:
    if default is not None and not os.path.exists(file_path):
        return default
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data


def save_obj(data: Any, file_path: str):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as f:
        pickle.dump(data, f)
