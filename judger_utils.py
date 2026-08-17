import logging
import sys
from pathlib import Path
from collections import defaultdict

from openai import OpenAI

from llm_utils import OPENAI_JUDGE_KEYS, load_yaml_file


def get_logger(prefix: str, log_dir: Path, stream: bool = True):
    logger = logging.getLogger(prefix)

    if logger.handlers:
        return logger

    log_dir.mkdir(exist_ok=True)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(message)s")

    file_handler = logging.FileHandler(log_dir / f"{prefix}.log", mode="w")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if stream:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    logger.propagate = False
    return logger


def get_judge_client(prefix: str) -> OpenAI:
    key = OPENAI_JUDGE_KEYS[prefix]
    if not key:
        raise RuntimeError(f"Missing judge API key for {prefix}")
    return OpenAI(api_key=key)


def get_openai_judge_key_from_path(path: Path, base_dir: Path) -> str:
    model_dir = path.relative_to(base_dir).parts[0].lower()
    for prefix, key in OPENAI_JUDGE_KEYS.items():
        if model_dir.startswith(prefix):
            if not key:
                raise RuntimeError(f"Missing OpenAI judge key for {prefix}")
            return key
    raise RuntimeError(f"Unknown model dir: {model_dir}")


def iter_json_files(base_dir: Path):
    yield from base_dir.rglob("run_*.json")


def group_files_by_model_prefix(base_dir: Path):
    groups = defaultdict(list)
    for path in iter_json_files(base_dir):
        model_dir = path.relative_to(base_dir).parts[0].lower()
        for prefix in OPENAI_JUDGE_KEYS:
            if model_dir.startswith(prefix):
                groups[prefix].append(path)
                break
        else:
            raise RuntimeError(f"Unknown model dir: {model_dir}")
    return groups


def gpt4o_complete(client: OpenAI, messages: list) -> str:
    return client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0,
    ).choices[0].message.content.strip().lower()
