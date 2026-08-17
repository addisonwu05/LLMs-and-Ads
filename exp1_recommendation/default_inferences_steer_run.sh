#!/bin/bash
export PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd):${PYTHONPATH:-}"

# ===== MODEL FAMILIES =====
gpt_models=("gpt-5.1" "gpt-5-mini")
claude_models=("claude-opus-4-5-20251101")
gemini_3_models=("gemini-3-pro-preview")
together_models=("meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8")
together_qwen_models=("Qwen/Qwen3-Next-80B-A3B-Thinking" "Qwen/Qwen3-Next-80B-A3B-Instruct")
together_deepseek_models=("deepseek-ai/DeepSeek-V3.1" "deepseek-ai/DeepSeek-R1")
grok_models=("grok-4-1-fast-reasoning")

# ===== EXPERIMENT CONDITIONS =====
swap_combos=("")
flag_combinations=("" "--direct")

persona_combinations=(
  "--disadvantaged_persona"
  ""
)

details_present=("")

# ===== STEERING CONDITIONS =====
steer_targets=("equality" "user" "company")

NUM_RUNS=100
OUTDIR="results"
mkdir -p "$OUTDIR"

# ===============================
# MAIN PER-MODEL RUNNER
# ===============================
run_all_conditions_for_model() {
  local model="$1"
  echo "=== Running model: $model ==="

  #for SYS_PROMPT in 1 2 3; do
  for SYS_PROMPT in 1; do
    echo "-------------------------------"
    echo "$model | SYSTEM PROMPT $SYS_PROMPT"
    echo "-------------------------------"

    for steer_target in "${steer_targets[@]}"; do
      for persona_flag in "${persona_combinations[@]}"; do
        for swap_flag in "${swap_combos[@]}"; do
          for reasoning_flag in "${flag_combinations[@]}"; do
            for details_flag in "${details_present[@]}"; do

              # Skip unsupported --direct cases
              if [[ "$reasoning_flag" == "--direct" && \
                    ( "$model" == "deepseek-ai/DeepSeek-R1" || \
                      "$model" == "Qwen/Qwen3-Next-80B-A3B-Thinking" || \
                      "$model" == "gemini-3-pro-preview" || \
                      "$model" == "grok-4-1-fast-reasoning") ]]; then
                echo "[-] Skipping --direct for $model"
                continue
              fi

              echo "[+] $model | STEER=$steer_target | persona=$persona_flag | mode=$reasoning_flag"

              python3 default_inferences.py \
                --model "$model" \
                --num_runs "$NUM_RUNS" \
                --system_prompt "$SYS_PROMPT" \
                --nonsponsored_flights_less_expensive \
                --steer \
                --steer_towards "$steer_target" \
                $swap_flag \
                $persona_flag \
                $reasoning_flag \
                $details_flag

            done
          done
        done
      done
    done

    # ===============================
    # HIGH REASONING PASS (GPT-5 ONLY)
    # ===============================
    if [[ "$model" == "gpt-5-mini" || "$model" == "gpt-5.1" ]]; then
      echo "=== HIGH reasoning pass for $model (SYS_PROMPT $SYS_PROMPT) ==="

      for steer_target in "${steer_targets[@]}"; do
        for persona_flag in "${persona_combinations[@]}"; do
          for swap_flag in "${swap_combos[@]}"; do
            for details_flag in "${details_present[@]}"; do

              echo "[+] $model | HIGH | STEER=$steer_target | persona=$persona_flag"

              python3 default_inferences.py \
                --model "$model" \
                --num_runs "$NUM_RUNS" \
                --system_prompt "$SYS_PROMPT" \
                --nonsponsored_flights_less_expensive \
                --reasoning_level "high" \
                --steer \
                --steer_towards "$steer_target" \
                $swap_flag \
                $persona_flag \
                $details_flag

            done
          done
        done
      done
    fi

  done
}

# ===============================
# FAMILY SERIAL RUNNER
# ===============================
run_family_serially() {
  local models=("$@")
  for model in "${models[@]}"; do
    run_all_conditions_for_model "$model"
  done
}

# ===============================
# PARALLEL ACROSS FAMILIES
# ===============================
(run_family_serially "${gpt_models[@]}") &
(run_family_serially "${claude_models[@]}") &
(run_family_serially "${gemini_3_models[@]}") &
(run_family_serially "${together_models[@]}") &
(run_family_serially "${together_qwen_models[@]}") &
(run_family_serially "${together_deepseek_models[@]}") &
(run_family_serially "${grok_models[@]}") &

# ===============================
# THINKING PASS (Claude Opus only)
# ===============================
run_thinking_steer_conditions() {
  local model="claude-opus-4-5-20251101"
  for SYS_PROMPT in 1; do
    for steer_target in "${steer_targets[@]}"; do
      for persona_flag in "${persona_combinations[@]}"; do
        for swap_flag in "${swap_combos[@]}"; do
          for details_flag in "${details_present[@]}"; do
            echo "[+] $model | THINKING | STEER=$steer_target | persona=$persona_flag"
            python3 default_inferences.py \
              --model "$model" \
              --num_runs "$NUM_RUNS" \
              --system_prompt "$SYS_PROMPT" \
              --nonsponsored_flights_less_expensive \
              --steer \
              --steer_towards "$steer_target" \
              --thinking \
              $swap_flag $persona_flag $details_flag
          done
        done
      done
    done
  done
}
(run_thinking_steer_conditions) &

wait
echo "All steered flight recommendation experiments complete."
