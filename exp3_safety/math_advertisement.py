import argparse
import random
import os
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_utils import init_client, init_judge_client, chat_completion, get_next_index, load_yaml_file

_HERE = Path(__file__).parent
prompts = load_yaml_file(_HERE / "prompts_math.yaml")

with open(_HERE / "hendrycks_math_all.json", "r") as f:
    ALL_PROBLEMS = json.load(f)


def make_output_dir(args):
    model_dir = args.model.replace("/", "_")
    if args.reasoning_level != "minimal":
        model_dir += args.reasoning_level
    if args.thinking:
        model_dir += "thinking"
    mode = "direct" if args.direct else "cot"
    path = os.path.join("math_advertisement_results", model_dir, mode)
    os.makedirs(path, exist_ok=True)
    return path


def simulate_recommendations_one_each(args, client):
    system_prompt = prompts["study_system_prompt_1"]
    message_log = [{"role": "system", "content": system_prompt}]

    ex = random.choice(ALL_PROBLEMS)
    math_problem = ex["problem"]

    request_prompt = prompts["student_context_prompt"].format(problem=math_problem)

    if args.direct:
        request_prompt += " " + prompts["direct_prompt_addon"]
    else:
        request_prompt += " " + prompts["cot_prompt_addon"]

    message_log.append({"role": "user", "content": request_prompt})
    model_response = chat_completion(args, client, message_log, max_tokens=1500, max_retries=5)
    message_log.append({"role": "assistant", "content": model_response})

    return message_log, math_problem, model_response


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--num_runs", type=int, default=30)
    parser.add_argument("--direct", action="store_true")
    parser.add_argument("--alter_temperature", action="store_true")
    parser.add_argument("--reasoning_level", type=str, default="minimal")
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--thinking_budget", type=int, default=1024)
    return parser.parse_args()


def main(args):
    client = init_client(args)
    judge_client = init_judge_client(args)
    outdir = make_output_dir(args)

    existing_files = [f for f in os.listdir(outdir) if f.startswith("run_") and f.endswith(".json")]
    existing_runs = len(existing_files)

    if existing_runs >= args.num_runs:
        print(f"[skip] {existing_runs} runs already exist (limit = {args.num_runs}).")
        return

    start_index = get_next_index(outdir)
    remaining = args.num_runs - existing_runs
    end_index = start_index + remaining

    print(f"Existing runs: {existing_runs}")
    print(f"Need to add: {remaining}")
    print(f"Creating runs {start_index} to {end_index - 1}")

    for run_idx in range(start_index, end_index):
        message_log, math_problem, model_response = simulate_recommendations_one_each(args, client)

        out_obj = {
            "math_problem": math_problem,
            "messages": message_log,
            "model_response": model_response,
        }

        out_path = os.path.join(outdir, f"run_{run_idx:03d}.json")
        with open(out_path, "w") as f:
            json.dump(out_obj, f, indent=2)

        print(f"[saved] {out_path}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
