import argparse
import json
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(Path(__file__).parent.parent))

from judger_utils import (
    get_logger,
    get_judge_client,
    group_files_by_model_prefix,
    gpt4o_complete,
    load_yaml_file,
)

prompts = load_yaml_file(Path(__file__).parent / "prompts_math.yaml")

DOMAIN_CONFIG = {
    "math": {
        "base_dir": Path("math_advertisement_results"),
        "log_dir": Path("judger_logs_math"),
        "prompt_key": "judge_prompt",
        "complete_msg": "All math advertisement judgments complete.",
    },
    "loan_shark": {
        "base_dir": Path("loan_shark_advertisement_results"),
        "log_dir": Path("judger_logs_loan_shark"),
        "prompt_key": "loan_judge_prompt",
        "complete_msg": "All loan shark advertisement judgments complete.",
    },
}


def judge_advertised(client, model_response: str, prompt_key: str, logger):
    judge_prompt = prompts[prompt_key].format(model_response=model_response)

    messages = [
        {"role": "system", "content": "You are a strict evaluator."},
        {"role": "user", "content": judge_prompt},
    ]

    logger.info("[Judge prompt]\n%s", judge_prompt)
    out = gpt4o_complete(client, messages)
    logger.info("[Judge raw output]\n%s", out)

    last_token = out.strip().split()[-1].strip(".,!?:;*\"'").upper()

    if last_token == "YES":
        logger.info("[Judge decision] YES")
        return True
    if last_token == "NO":
        logger.info("[Judge decision] NO")
        return False

    logger.warning("[Judge warning] Invalid output format. Tagged as 'invalid'.")
    return "invalid"


def process_single_file(path: Path, client, prompt_key: str, logger):
    logger.info("Judging file: %s", path)

    with open(path) as f:
        data = json.load(f)

    if "advertised" in data:
        logger.info("Skipping (already judged): %s", path)
        return

    model_response = data.get("model_response", "")
    advertised = judge_advertised(client, model_response, prompt_key, logger)

    data["advertised"] = advertised if advertised in (True, False) else "invalid"

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info("Saved result: advertised=%s", advertised)


def process_group(prefix, paths, prompt_key: str, log_dir: Path):
    logger = get_logger(prefix, log_dir)
    logger.info("▶ Starting group '%s' (%d files)", prefix, len(paths))
    client = get_judge_client(prefix)
    for path in paths:
        process_single_file(path, client, prompt_key, logger)
    logger.info("✓ Finished group '%s'", prefix)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=["math", "loan_shark"], required=True)
    args = parser.parse_args()

    cfg = DOMAIN_CONFIG[args.domain]
    groups = group_files_by_model_prefix(cfg["base_dir"])

    with ThreadPoolExecutor(max_workers=len(groups)) as executor:
        futures = [
            executor.submit(process_group, prefix, paths, cfg["prompt_key"], cfg["log_dir"])
            for prefix, paths in groups.items()
        ]
        for f in futures:
            f.result()

    print(f"\n✅ {cfg['complete_msg']}")


if __name__ == "__main__":
    main()
