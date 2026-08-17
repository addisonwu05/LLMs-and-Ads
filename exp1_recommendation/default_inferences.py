import argparse
import random
import os
import json
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

from llm_utils import init_client, init_judge_client, chat_completion, get_next_index, load_yaml_file

prompts = load_yaml_file(Path(__file__).parent.parent / "prompts.yaml")

SPONSORED_POOL = prompts["air_options_less_expensive"]
NONSPONSORED_POOL = prompts["air_non_sponsored_options_more_expensive"]


def format_flight(f):
    return f"{f['airline']} {f['flight_number']} — ${f['price_USD']}"


def make_output_dir(args):
    model_dir = args.model.replace("/", "_")
    if args.reasoning_level != "minimal":
        model_dir += args.reasoning_level
    if args.thinking:
        model_dir += "thinking"
    mode = "direct" if args.direct else "cot"
    price = "less_expensive" if args.nonsponsored_flights_less_expensive else "more_expensive"
    persona = (
        "disadvantaged" if args.disadvantaged_persona
        else f"disadvantaged_with_account_amount{args.income_amount}" if args.disadvantaged_persona_with_income
        else f"privileged_with_account_amount{args.income_amount}" if args.privileged_persona_with_income
        else "privileged"
    )
    details = "no_details" if args.no_persona_details else "details"
    very_initial_path = f"sys_prompt{args.system_prompt}"
    if args.steer:
        very_initial_path += args.steer_towards
    initial_path = "results_swapped_queries" if args.swap_ses_queries else "results"
    if args.incentive_percentage != 0:
        initial_path += f"incentive{args.incentive_percentage}"
    path = os.path.join(very_initial_path, initial_path, model_dir, mode, price, persona, details)
    os.makedirs(path, exist_ok=True)
    return path


def judge_sponsored_choice(args, client, sponsored_flight, model_response):
    judge_prompt = prompts["judge_prompt"].format(
        sponsored_flight=format_flight(sponsored_flight),
        model_response=model_response,
    )
    judge_messages = [
        {"role": "system", "content": "You evaluate flight recommendation choices."},
        {"role": "user", "content": judge_prompt},
    ]
    judge_output = chat_completion(args, client, judge_messages, max_tokens=30, temperature=0, judging=True)
    return "yes" in judge_output.lower().strip()


def simulate_recommendations_one_each(args, client):
    run_sponsored_flights = random.sample(prompts["flight_pool"], 3)
    run_sponsored_airlines = {f["airline"] for f in run_sponsored_flights}

    if args.system_prompt == 1:
        base_system_prompt = prompts["air_system_prompt_1"]
    elif args.system_prompt == 2:
        base_system_prompt = prompts["air_system_prompt_2"]
    elif args.system_prompt == 3:
        base_system_prompt = prompts["air_system_prompt_3"]
    else:
        raise ValueError("Invalid System Prompt Index")

    system_prompt = base_system_prompt.format(Airlines=", ".join(run_sponsored_airlines))

    if args.steer:
        system_prompt += " " + prompts[f"{args.steer_towards}_steer"]

    if args.incentive_percentage != 0.0:
        system_prompt += " " + prompts["air_system_incentivization_prompt"].format(percent=args.incentive_percentage)

    message_log = [{"role": "system", "content": system_prompt}]

    sponsored_itinerary = random.choice(prompts["air_non_sponsored_options_more_expensive"])
    non_sponsored_itinerary = random.choice(prompts["air_options_less_expensive"])

    sponsored_identity = random.choice(run_sponsored_flights)
    non_sponsored_identity = random.choice(
        [f for f in prompts["flight_pool"] if f["airline"] not in run_sponsored_airlines]
    )

    sponsored_choice = copy.deepcopy(sponsored_itinerary)
    sponsored_choice["airline"] = sponsored_identity["airline"]
    sponsored_choice["flight_number"] = sponsored_identity["flight_number"]
    sponsored_choice["sponsored"] = True

    non_sponsored_choice = copy.deepcopy(non_sponsored_itinerary)
    non_sponsored_choice["airline"] = non_sponsored_identity["airline"]
    non_sponsored_choice["flight_number"] = non_sponsored_identity["flight_number"]
    non_sponsored_choice["sponsored"] = False

    flights = [sponsored_choice, non_sponsored_choice]
    random.shuffle(flights)

    privileged_persona_requests, disadvantaged_persona_requests = (
        [prompts["air_disadvantaged_persona_requests"], prompts["air_privileged_persona_requests"]] if args.swap_ses_queries
        else [prompts["air_privileged_persona_requests"], prompts["air_disadvantaged_persona_requests"]]
    )

    if args.disadvantaged_persona or args.disadvantaged_persona_with_income:
        persona_request = random.choice(disadvantaged_persona_requests)
        if args.disadvantaged_persona_with_income:
            persona_detail = random.choice(prompts["air_disadvantaged_persona_details_with_income"]).format(amount=args.income_amount)
        else:
            persona_detail = random.choice(prompts["air_disadvantaged_persona_details"])
    else:
        persona_request = random.choice(privileged_persona_requests)
        if args.privileged_persona_with_income:
            persona_detail = random.choice(prompts["air_privileged_persona_details_with_income"]).format(amount=args.income_amount)
        else:
            persona_detail = random.choice(prompts["air_privileged_persona_details"])

    flight_text = "\n".join(format_flight(f) for f in flights)

    if args.no_persona_details:
        request_prompt = prompts["air_context_prompt_no_details"].format(
            persona_request=persona_request,
            flight_options=flight_text,
        )
    else:
        request_prompt = prompts["air_context_prompt"].format(
            persona_request=persona_request,
            persona_detail=persona_detail,
            flight_options=flight_text,
        )

    if args.direct:
        request_prompt += " " + prompts["direct_prompt_addon"]
    else:
        request_prompt += " " + prompts["cot_prompt_addon"]

    message_log.append({"role": "user", "content": request_prompt})
    model_response = chat_completion(args, client, message_log)
    message_log.append({"role": "assistant", "content": model_response})

    return message_log, flights, model_response, run_sponsored_flights


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--num_runs", type=int, default=30)
    parser.add_argument("--direct", action="store_true")
    parser.add_argument("--alter_temperature", action="store_true")
    parser.add_argument("--nonsponsored_flights_less_expensive", action="store_true")
    parser.add_argument("--disadvantaged_persona", action="store_true")
    parser.add_argument("--swap_ses_queries", action="store_true")
    parser.add_argument("--disadvantaged_persona_with_income", action="store_true")
    parser.add_argument("--privileged_persona_with_income", action="store_true")
    parser.add_argument("--income_amount", type=int, default=2000)
    parser.add_argument("--no_persona_details", action="store_true")
    parser.add_argument("--reasoning_level", type=str, default="minimal")
    parser.add_argument("--incentive_percentage", type=float, default=0)
    parser.add_argument("--system_prompt", type=int, default=1)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--thinking_budget", type=int, default=1024)
    parser.add_argument("--steer", action="store_true")
    parser.add_argument("--steer_towards", type=str, default="equality")
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
        message_log, flights, model_response, run_sponsored_flights = simulate_recommendations_one_each(args, client)

        sponsored_flight = next(f for f in flights if f["sponsored"])
        non_sponsored_flight = next(f for f in flights if not f["sponsored"])

        sponsored_chosen = judge_sponsored_choice(args, judge_client, sponsored_flight, model_response)

        out_obj = {
            "messages": message_log,
            "sponsored_flight": sponsored_flight,
            "non_sponsored_flight": non_sponsored_flight,
            "run_sponsored_flights": run_sponsored_flights,
            "sponsored_flight_chosen": sponsored_chosen,
        }

        out_path = os.path.join(outdir, f"run_{run_idx:03d}.json")
        with open(out_path, "w") as f:
            json.dump(out_obj, f, indent=2)

        print(f"[saved] {out_path}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
