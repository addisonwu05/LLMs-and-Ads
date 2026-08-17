#!/bin/bash
export PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd):${PYTHONPATH:-}"

# ===== MODEL FAMILIES =====
gpt_models=("gpt-5.1" "gpt-5-mini")
gpt_old_models=("gpt-4o" "gpt-3.5-turbo-0125")
claude_models=("claude-opus-4-5-20251101" "claude-sonnet-4-20250514" "claude-3-haiku-20240307")
gemini_models=("gemini-2.5-flash" "gemini-2.0-flash")
gemini_3_models=("gemini-3-pro-preview")
together_models=(
  "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
  "meta-llama/Llama-4-Scout-17B-16E-Instruct"
  "meta-llama/Llama-3.3-70B-Instruct-Turbo"
  "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
)
together_qwen_models=(
  "Qwen/Qwen3-Next-80B-A3B-Thinking"
  "Qwen/Qwen3-235B-A22B-Instruct-2507-tput"
  "Qwen/Qwen3-Next-80B-A3B-Instruct"
  "Qwen/Qwen2.5-VL-72B-Instruct"
  "Qwen/Qwen2.5-7B-Instruct-Turbo"
)
together_deepseek_models=("deepseek-ai/DeepSeek-V3.1" "deepseek-ai/DeepSeek-V3" "deepseek-ai/DeepSeek-R1")
grok_models=("grok-4-1-fast-non-reasoning" "grok-4-1-fast-reasoning" "grok-4-fast-non-reasoning")
grok_3_models=("grok-3-mini" "grok-3")

# ===== EXPERIMENT CONDITIONS =====
swap_combos=(
  ""
  #"--swap_ses_queries"
)

flag_combinations=(
  ""
  "--direct"
)

persona_combinations=(
  "--disadvantaged_persona_with_income"
  #"--disadvantaged_persona"
  "--privileged_persona_with_income"
  #""
)

details_present=(
  ""
)

#NUM_RUNS=100
NUM_RUNS=50
OUTDIR="results"
mkdir -p "$OUTDIR"

# ===============================
# MAIN PER-MODEL RUNNER
# ===============================
run_all_conditions_for_model() {
  local model="$1"
  echo "=== Running model: $model ==="

  for SYS_PROMPT in 1 2 3; do
    echo "-------------------------------"
    echo "🧠 $model | SYSTEM PROMPT $SYS_PROMPT"
    echo "-------------------------------"

    #if [[ "$SYS_PROMPT" == 2 || "$SYS_PROMPT" == 3 ]]; then
    #  persona_combinations=("--disadvantaged_persona" "")
    #fi

    # ===== STANDARD PASSES =====
    for persona_flag in "${persona_combinations[@]}"; do
      # Only allow system_prompt=1 for income personas
      if [[ "$persona_flag" == *"_with_income" && "$SYS_PROMPT" != "1" ]]; then
        continue
      fi

      if [[ "$persona_flag" == "--privileged_persona_with_income" ]]; then
        incentive_values=("" "1")
      else
        incentive_values=("" "1" "10" "20")
      fi

      if [[ "$SYS_PROMPT" == 2 || "$SYS_PROMPT" == 3 ]]; then
        incentive_values=("")
      fi

      for incentive in "${incentive_values[@]}"; do
        if [[ -n "$incentive" && "$SYS_PROMPT" != "1" ]]; then
          continue
        fi
        for swap_flag in "${swap_combos[@]}"; do
          for reasoning_flag in "${flag_combinations[@]}"; do

            # Skip unsupported direct reasoning cases
            if [[ "$reasoning_flag" == "--direct" && \
                  ( "$model" == "deepseek-ai/DeepSeek-R1" || "$model" == "Qwen/Qwen3-Next-80B-A3B-Thinking" || "$model" == "gemini-3-pro-preview" || "$model" == "claude-opus-4-5-20251101" ) ]]; then
              echo "[-] Skipping --direct for $model"
              continue
            fi

            for details_flag in "${details_present[@]}"; do

              if [[ "$persona_flag" == "--disadvantaged_persona_with_income" ]]; then
                income_values=("400" "1000" "2000" "5000" "10000")
              elif [[ "$persona_flag" == "--privileged_persona_with_income" ]]; then
                income_values=("400" "1000" "2000" "20000" "200000")
              else
                income_values=("")
              fi

              #if [[ "$SYS_PROMPT" == 2 || "$SYS_PROMPT" == 3 ]]; then
              #  income_values=("")
              #fi

              for income in "${income_values[@]}"; do

                echo "[+] $model | incentive=$incentive | swap=$swap_flag | persona=$persona_flag | reasoning=$reasoning_flag | income=$income"

                if [[ -n "$income" ]]; then
                  python3 default_inferences.py \
                    --model "$model" \
                    --num_runs "$NUM_RUNS" \
                    --system_prompt "$SYS_PROMPT" \
                    --nonsponsored_flights_less_expensive \
                    ${incentive:+--incentive_percentage "$incentive"} \
                    $swap_flag \
                    $persona_flag $reasoning_flag $details_flag \
                    --income_amount "$income"
                else
                  python3 default_inferences.py \
                    --model "$model" \
                    --num_runs "$NUM_RUNS" \
                    --system_prompt "$SYS_PROMPT" \
                    --nonsponsored_flights_less_expensive \
                    ${incentive:+--incentive_percentage "$incentive"} \
                    $swap_flag \
                    $persona_flag $reasoning_flag $details_flag
                fi

              done
            done
          done
        done
      done
    done

    # ===== HIGH-REASONING PASS (GPT-5 ONLY) =====
    if [[ "$model" == "gpt-5-mini" || "$model" == "gpt-5.1" ]]; then
      echo "=== Running HIGH-REASONING mode for $model ==="

      for persona_flag in "${persona_combinations[@]}"; do
        # Only allow system_prompt=1 for income personas
        if [[ "$persona_flag" == *"_with_income" && "$SYS_PROMPT" != "1" ]]; then
          continue
        fi

        if [[ "$persona_flag" == "--privileged_persona_with_income" ]]; then
          incentive_values=("" "1")
        else
          incentive_values=("" "1" "10" "20")
        fi

        if [[ "$SYS_PROMPT" == 2 || "$SYS_PROMPT" == 3 ]]; then
          incentive_values=("")
        fi

        for incentive in "${incentive_values[@]}"; do
        if [[ -n "$incentive" && "$SYS_PROMPT" != "1" ]]; then
          continue
        fi
          for swap_flag in "${swap_combos[@]}"; do
            for details_flag in "${details_present[@]}"; do

              if [[ "$persona_flag" == "--disadvantaged_persona_with_income" ]]; then
                income_values=("400" "1000" "2000" "5000" "10000")
              elif [[ "$persona_flag" == "--privileged_persona_with_income" ]]; then
                income_values=("400" "1000" "2000" "20000" "200000")
              else
                income_values=("")
              fi

              #if [[ "$SYS_PROMPT" == 2 || "$SYS_PROMPT" == 3 ]]; then
              #  income_values=("")
              #fi

              for income in "${income_values[@]}"; do

                echo "[+] $model | HIGH | incentive=$incentive | swap=$swap_flag | persona=$persona_flag | income=$income"

                if [[ -n "$income" ]]; then
                  python3 default_inferences.py \
                    --model "$model" \
                    --num_runs "$NUM_RUNS" \
                    --system_prompt "$SYS_PROMPT" \
                    --nonsponsored_flights_less_expensive \
                    --reasoning_level "high" \
                    ${incentive:+--incentive_percentage "$incentive"} \
                    $swap_flag \
                    $persona_flag $details_flag \
                    --income_amount "$income"
                else
                  python3 default_inferences.py \
                    --model "$model" \
                    --num_runs "$NUM_RUNS" \
                    --system_prompt "$SYS_PROMPT" \
                    --nonsponsored_flights_less_expensive \
                    --reasoning_level "high" \
                    ${incentive:+--incentive_percentage "$incentive"} \
                    $swap_flag \
                    $persona_flag $details_flag
                fi

              done
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
(run_family_serially "${gpt_old_models[@]}") &
(run_family_serially "${claude_models[@]}") &
(run_family_serially "${gemini_models[@]}") &
(run_family_serially "${gemini_3_models[@]}") &
(run_family_serially "${together_models[@]}") &
(run_family_serially "${together_qwen_models[@]}") &
(run_family_serially "${together_deepseek_models[@]}") &
(run_family_serially "${grok_models[@]}") &
(run_family_serially "${grok_3_models[@]}") &

# ===============================
# THINKING PASS (Claude Opus only)
# ===============================
run_thinking_conditions() {
  local model="claude-opus-4-5-20251101"
  local thinking_personas=(
    "--disadvantaged_persona_with_income"
    "--privileged_persona_with_income"
    "--disadvantaged_persona"
    ""
  )

  for SYS_PROMPT in 1 2 3; do
    for persona_flag in "${thinking_personas[@]}"; do
      if [[ "$persona_flag" == *"_with_income" && "$SYS_PROMPT" != "1" ]]; then
        continue
      fi

      if [[ "$persona_flag" == "--privileged_persona_with_income" ]]; then
        incentive_values=("" "1")
      else
        incentive_values=("" "1" "10" "20")
      fi
      if [[ "$SYS_PROMPT" == 2 || "$SYS_PROMPT" == 3 ]]; then
        incentive_values=("")
      fi

      for incentive in "${incentive_values[@]}"; do
        for swap_flag in "${swap_combos[@]}"; do

          if [[ "$persona_flag" == "--disadvantaged_persona_with_income" ]]; then
            income_values=("400" "1000" "2000" "5000" "10000")
          elif [[ "$persona_flag" == "--privileged_persona_with_income" ]]; then
            income_values=("400" "1000" "2000" "20000" "200000")
          else
            income_values=("")
          fi

          for income in "${income_values[@]}"; do
            echo "[+] $model | THINKING | SYS=$SYS_PROMPT | persona=$persona_flag | incentive=$incentive | income=$income"
            python3 default_inferences.py \
              --model "$model" \
              --num_runs "$NUM_RUNS" \
              --system_prompt "$SYS_PROMPT" \
              --nonsponsored_flights_less_expensive \
              --thinking \
              ${incentive:+--incentive_percentage "$incentive"} \
              $swap_flag \
              $persona_flag \
              ${income:+--income_amount "$income"}
          done
        done
      done
    done
  done
}
(run_thinking_conditions) &

wait
echo "✅ All flight recommendation experiments complete."