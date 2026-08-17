#!/bin/bash
export PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd):${PYTHONPATH:-}"

# ================================
# MODEL FAMILIES
# ================================
gpt_models=("gpt-5.1" "gpt-5-mini")
gpt_old_models=("gpt-4o" "gpt-3.5-turbo-0125")
claude_models=("claude-opus-4-5-20251101" "claude-sonnet-4-20250514" "claude-3-haiku-20240307")
gemini_models=("gemini-2.5-flash" "gemini-2.0-flash")
gemini_3_models=("gemini-3-pro-preview")
together_models=(
  "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
  "meta-llama/Llama-3.3-70B-Instruct-Turbo"
  "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
)
together_qwen_models=(
  "Qwen/Qwen3-Next-80B-A3B-Instruct"
  "Qwen/Qwen2.5-VL-72B-Instruct"
  "Qwen/Qwen2.5-7B-Instruct-Turbo"
)
together_deepseek_models=(
  "deepseek-ai/DeepSeek-R1"
  "deepseek-ai/DeepSeek-V3.1"
  "deepseek-ai/DeepSeek-V3"
)
grok_models=("grok-4-1-fast-non-reasoning" "grok-4-1-fast-reasoning" "grok-4-fast-non-reasoning")
grok_3_models=("grok-3-mini" "grok-3")

# ================================
# EXPERIMENT CONDITIONS
# ================================
swap_combos=(
    ""
    #"--swap_ses_queries"
)

flag_combinations=(
    ""
    "--direct"
)

persona_combinations=(
    #"--disadvantaged_persona_with_income"
    "--disadvantaged_persona"
    ""
)

details_present=(
    #"--no_persona_details"
    ""
)

num_nonsponsored_flight_options=(
    ""
    #"--num_nonsponsored_flight_options 2"
)

NUM_RUNS=100
OUTDIR="results"
mkdir -p "$OUTDIR"

# ================================
# HELPERS
# ================================
timestamp() {
    date +"%Y%m%d_%H%M%S"
}

# ================================
# RUN ALL CONDITIONS FOR ONE MODEL
# ================================
run_all_conditions_for_model() {
    local model="$1"
    echo "=== Running model: $model ==="

    for swap_flag in "${swap_combos[@]}"; do
      for persona_flag in "${persona_combinations[@]}"; do
        for reasoning_flag in "${flag_combinations[@]}"; do

        if [[ "$model" == "deepseek-ai/DeepSeek-R1" && "$reasoning_flag" == "--direct" ]]; then
            continue
        fi
          for details_flag in "${details_present[@]}"; do
            for ns_flag in "${num_nonsponsored_flight_options[@]}"; do

              # ----------------
              # Income logic
              # ----------------
              income_values=("")

              for income in "${income_values[@]}"; do

                # ----------------
                # Labels
                # ----------------
                persona_label="priv"
                [[ "$persona_flag" == *"--disadvantaged_persona"* ]] && persona_label="disadv"

                reasoning_label="cot"
                [[ "$reasoning_flag" == *"--direct"* ]] && reasoning_label="direct"

                details_label="details"
                [[ "$details_flag" == *"--no_persona_details"* ]] && details_label="noDetails"

                swap_label=""
                [[ "$swap_flag" == *"--swap_ses_queries"* ]] && swap_label="_swap"

                income_label=""
                [[ -n "$income" ]] && income_label="_inc${income}"

                ns_label=""
                if [[ "$ns_flag" == *"--num_nonsponsored_flight_options"* ]]; then
                    ns_label="_ns$(echo "$ns_flag" | grep -o '[0-9]\+')"
                fi

                run_tag="${persona_label}_${reasoning_label}_${details_label}${swap_label}${income_label}${ns_label}"

                echo "[+] $model : $run_tag"

                # ----------------
                # Command
                # ----------------
                if [[ -n "$income" ]]; then
                    python3 surfacing_inferences.py \
                        --model "$model" \
                        --num_runs "$NUM_RUNS" \
                        --nonsponsored_flights_less_expensive \
                        $ns_flag \
                        $swap_flag \
                        $persona_flag $reasoning_flag $details_flag \
                        --income_amount "$income"
                else
                    python3 surfacing_inferences.py \
                        --model "$model" \
                        --num_runs "$NUM_RUNS" \
                        --nonsponsored_flights_less_expensive \
                        $ns_flag \
                        $swap_flag \
                        $persona_flag $reasoning_flag $details_flag
                fi

              done
            done
          done
        done
      done
    done

    # ================================
    # HIGH-REASONING GPT RUNS
    # ================================
    if [[ "$model" == "gpt-5-mini" || "$model" == "gpt-5.1" ]]; then
        echo "=== Running extra HIGH-REASONING mode for $model ==="

        for swap_flag in "${swap_combos[@]}"; do
          for persona_flag in "${persona_combinations[@]}"; do
            for details_flag in "${details_present[@]}"; do
              for ns_flag in "${num_nonsponsored_flight_options[@]}"; do

                #if [[ "$persona_flag" == "--disadvantaged_persona_with_income" ]]; then
                #    income_values=("2000" "5000" "10000")
                #else
                #    income_values=("")
                #fi
                income_values=("")

                for income in "${income_values[@]}"; do

                  persona_label="priv"
                  [[ "$persona_flag" == *"--disadvantaged_persona"* ]] && persona_label="disadv"

                  reasoning_label="highreason"

                  details_label="details"
                  [[ "$details_flag" == *"--no_persona_details"* ]] && details_label="noDetails"

                  swap_label=""
                  [[ "$swap_flag" == *"--swap_ses_queries"* ]] && swap_label="_swap"

                  income_label=""
                  [[ -n "$income" ]] && income_label="_inc${income}"

                  ns_label=""
                  if [[ "$ns_flag" == *"--num_nonsponsored_flight_options"* ]]; then
                      ns_label="_ns$(echo "$ns_flag" | grep -o '[0-9]\+')"
                  fi

                  run_tag="${persona_label}_${reasoning_label}_${details_label}${swap_label}${income_label}${ns_label}"

                  echo "[+] $model : $run_tag (HIGH REASONING)"

                  if [[ -n "$income" ]]; then
                      python3 surfacing_inferences.py \
                          --model "$model" \
                          --num_runs "$NUM_RUNS" \
                          --nonsponsored_flights_less_expensive \
                          --reasoning_level "high" \
                          $ns_flag \
                          $swap_flag \
                          $persona_flag $details_flag \
                          --income_amount "$income"
                  else
                      python3 surfacing_inferences.py \
                          --model "$model" \
                          --num_runs "$NUM_RUNS" \
                          --nonsponsored_flights_less_expensive \
                          --reasoning_level "high" \
                          $ns_flag \
                          $swap_flag \
                          $persona_flag $details_flag
                  fi

                done
              done
            done
          done
        done
    fi
}

# ================================
# FAMILY SERIAL RUNNER
# ================================
run_family_serially() {
  local models=("$@")
  for model in "${models[@]}"; do
    run_all_conditions_for_model "$model"
  done
}

# ================================
# PARALLEL ACROSS MODEL FAMILIES
# ================================
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

# ================================
# THINKING PASS (Claude Opus only, CoT only)
# ================================
run_thinking_conditions() {
  local model="claude-opus-4-5-20251101"
  local thinking_personas=("--disadvantaged_persona" "")

  for swap_flag in "${swap_combos[@]}"; do
    for persona_flag in "${thinking_personas[@]}"; do
      for details_flag in "${details_present[@]}"; do
        for ns_flag in "${num_nonsponsored_flight_options[@]}"; do
          echo "[+] $model (thinking) : persona=$persona_flag"
          python3 surfacing_inferences.py \
            --model "$model" \
            --num_runs "$NUM_RUNS" \
            --nonsponsored_flights_less_expensive \
            --thinking \
            $ns_flag $swap_flag \
            $persona_flag $details_flag
        done
      done
    done
  done
}
(run_thinking_conditions) &

wait

echo "✅ All flight recommendation experiments complete."
