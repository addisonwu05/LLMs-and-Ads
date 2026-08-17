#!/bin/bash
export PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd):${PYTHONPATH:-}"
set -e

# ================================
# MODELS (UNCHANGED)
# ================================
gpt_models=("gpt-5.1" "gpt-5-mini")
#gpt_old_models=("gpt-4o" "gpt-3.5-turbo-0125")
claude_models=(
  "claude-opus-4-5-20251101"
#  "claude-sonnet-4-20250514"
#  "claude-3-haiku-20240307"
)
gemini_models=("gemini-2.5-flash" "gemini-2.0-flash")
gemini_3_models=("gemini-3-pro-preview")
together_models=(
  "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
#  "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
)
together_qwen_models=(
  "Qwen/Qwen3-Next-80B-A3B-Instruct"
  #"Qwen/Qwen2.5-VL-72B-Instruct"
  #"Qwen/Qwen2.5-7B-Instruct-Turbo"
)
together_deepseek_models=(
  "deepseek-ai/DeepSeek-V3.1"
#  "deepseek-ai/DeepSeek-V3"
)
grok_models=(
  #"grok-4-1-fast-non-reasoning"
  "grok-4-1-fast-reasoning"
  #"grok-4-fast-non-reasoning"
)
#grok_3_models=("grok-3-mini" "grok-3")

# ================================
# SETTINGS
# ================================
NUM_RUNS=100

REASONING_FLAGS=(
  ""
  "--direct"
)

# ================================
# RUNNER
# ================================
run_models() {
  local models=("$@")
  for model in "${models[@]}"; do
    for reasoning_flag in "${REASONING_FLAGS[@]}"; do

      label="cot"
      [[ "$reasoning_flag" == "--direct" ]] && label="direct"

      echo "[launch] $model | $label"

      python3 math_advertisement.py \
        --model "$model" \
        --num_runs "$NUM_RUNS" \
        $reasoning_flag &

    done
  done
}

# ================================
# PARALLEL BY FAMILY
# ================================
(run_models "${gpt_models[@]}") &
#(run_models "${gpt_old_models[@]}") &
(run_models "${claude_models[@]}") &
(run_models "${gemini_models[@]}") &
(run_models "${gemini_3_models[@]}") &
(run_models "${together_models[@]}") &
(run_models "${together_qwen_models[@]}") &
(run_models "${together_deepseek_models[@]}") &
(run_models "${grok_models[@]}") &
#(run_models "${grok_3_models[@]}") &

# ================================
# THINKING PASS (Claude Opus only)
# ================================
run_models_thinking() {
  local models=("$@")
  for model in "${models[@]}"; do
    echo "[launch] $model | thinking"
    python3 math_advertisement.py \
      --model "$model" \
      --num_runs "$NUM_RUNS" \
      --thinking &
  done
}
(run_models_thinking "claude-opus-4-5-20251101") &

wait
echo "✅ All parallel math runs complete."
