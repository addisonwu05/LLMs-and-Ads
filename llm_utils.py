import os
import re

from openai import OpenAI
from together import Together
from mistralai import Mistral
import anthropic
from dotenv import load_dotenv
import yaml

load_dotenv()

OpenAI_API_KEY = os.getenv("OpenAI_API_KEY")
Anthropic_API_KEY = os.getenv("Anthropic_API_KEY")
Gemini_API_KEY = os.getenv("Gemini_API_KEY")
Together_API_KEY = os.getenv("Together_API_KEY")
Grok_API_KEY = os.getenv("Grok_API_KEY")
Mistral_API_KEY = os.getenv("Mistral_API_KEY")

OPENAI_JUDGE_KEYS = {
    "gpt": OpenAI_API_KEY,
    "claude": OpenAI_API_KEY,
    "gemini": OpenAI_API_KEY,
    "grok": OpenAI_API_KEY,
    "meta": OpenAI_API_KEY,
    "qwen": OpenAI_API_KEY,
    "deepseek": OpenAI_API_KEY,
}


def get_next_index(path):
    existing = [f for f in os.listdir(path) if f.startswith("run_") and f.endswith(".json")]
    if not existing:
        return 1
    nums = [int(re.findall(r"run_(\d+)\.json", f)[0]) for f in existing]
    return max(nums) + 1


def load_yaml_file(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


def init_client(args):
    if "gpt" in args.model or args.model[0] == "o":
        return OpenAI(api_key=OpenAI_API_KEY)
    elif "gemini" in args.model:
        return OpenAI(api_key=Gemini_API_KEY, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
    elif "grok" in args.model:
        return OpenAI(api_key=Grok_API_KEY, base_url="https://api.x.ai/v1")
    elif "claude" in args.model:
        return anthropic.Anthropic(api_key=Anthropic_API_KEY)
    elif "mistral" in args.model:
        return Mistral(api_key=Mistral_API_KEY)
    else:
        return Together(api_key=Together_API_KEY)


def init_judge_client(args):
    model_lower = args.model.lower()
    for key, api_key in OPENAI_JUDGE_KEYS.items():
        if key in model_lower:
            return OpenAI(api_key=api_key)


def chat_completion(args, client, messages, max_tokens=500, temperature=1, judging=False, max_retries=1):
    if args.alter_temperature:
        temperature = args.temperature

    if judging:
        return client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=temperature,
        ).choices[0].message.content

    if "claude" in args.model:
        for attempt in range(max_retries):
            system_prompt = None
            filtered_messages = []
            for m in messages:
                if m["role"] == "system":
                    system_prompt = m["content"]
                else:
                    filtered_messages.append(m)

            request_kwargs = dict(
                model=args.model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=filtered_messages,
            )

            if args.thinking:
                request_kwargs["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": args.thinking_budget,
                }
                request_kwargs["max_tokens"] = args.thinking_budget + 1

            response = client.messages.create(**request_kwargs)

            for block in response.content:
                if block.type == "text":
                    return block.text

            print(f"[retry] Claude returned no text block (attempt {attempt+1}/{max_retries})")

        print("[warning] Claude failed to return text after retries")
        return None

    if "mistral" in args.model:
        return client.chat.complete(
            model=args.model,
            messages=messages,
        ).choices[0].message.content

    if args.model in ("gpt-5-mini", "gpt-5"):
        if args.direct:
            return client.responses.create(
                model=args.model,
                input=messages,
                reasoning={"effort": args.reasoning_level},
            ).output_text
        else:
            return client.responses.create(
                model=args.model,
                input=messages,
            ).output_text

    return client.chat.completions.create(
        model=args.model,
        messages=messages,
        temperature=temperature,
    ).choices[0].message.content
