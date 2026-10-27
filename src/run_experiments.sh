#!/bin/bash
# run_experiments.sh
# This script runs experiments in a combinatorial fashion.
# It accepts the following parameters:
#   --src-dataset (-sd) : comma-separated list of source datasets
#   --trg-dataset (-td) : comma-separated list of target datasets
#   --seed (-s)         : comma-separated list of seeds or an interval (e.g., 0-100)
#   --approach (-a)     : comma-separated list of approaches
#   --is-flat (-f)      : flag indicating flat structure (if present, add to command)
#   --cpu (-c)          : integer number of cores to use
#   --extra-args (-e)   : extra arguments to pass directly to "python main.py"
#   --log-keyword (-lk) : additional string for log_dir

usage() {
    echo "Usage: $0 --src-dataset <sd1,sd2,...> --trg-dataset <td1,td2,...>"
    echo "          --seed <s1,s2,... or X-Y>"
    echo "          --approach <appr1,appr2,...> [--is-flat] --cpu <num_cores>"
    echo "          [--log-keyword <keyword>] [--extra-args \"<args>\"]"
    exit 1
}

# Default values
SRC_DATASETS=""
TRG_DATASETS=""
SEEDS=""
APPROACHES=""
IS_FLAT=0
CPU_CORES=""
EXTRA_ARGS=""
LOG_KEYWORD=""

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -sd|--src-dataset)
            SRC_DATASETS="$2"
            shift
            ;;
        -td|--trg-dataset)
            TRG_DATASETS="$2"
            shift
            ;;
        -s|--seed)
            SEEDS="$2"
            shift
            ;;
        -a|--approach)
            APPROACHES="$2"
            shift
            ;;
        -f|--is-flat)
            IS_FLAT=1
            ;;
        -c|--cpu)
            CPU_CORES="$2"
            shift
            ;;
        -e|--extra-args)
            EXTRA_ARGS="$2"
            shift
            ;;
        -lk|--log-keyword)
            LOG_KEYWORD="$2"
            shift
            ;;
        *)
            echo "Unknown parameter passed: $1"
            usage
            ;;
    esac
    shift
done

# Verify required parameters
if [[ -z "$SRC_DATASETS" || -z "$TRG_DATASETS" || -z "$SEEDS" || -z "$APPROACHES" || -z "$CPU_CORES" ]]; then
    usage
fi

# Set flag variable for --is-flat
if [ "$IS_FLAT" -eq 1 ]; then
    FLAT_FLAG=" --is-flat"
else
    FLAT_FLAG=""
fi

# Split comma-separated lists into arrays
IFS=',' read -ra SRC_DATASET_ARR <<< "$SRC_DATASETS"
IFS=',' read -ra TRG_DATASET_ARR <<< "$TRG_DATASETS"
IFS=',' read -ra APPROACH_ARR <<< "$APPROACHES"

# Process the --seed parameter:
# If it matches the pattern X-Y, expand the range; otherwise, split by commas.
SEED_ARR=()
if [[ "$SEEDS" =~ ^[0-9]+-[0-9]+$ ]]; then
    IFS='-' read -r start end <<< "$SEEDS"
    for ((i=start; i<=end; i++)); do
        SEED_ARR+=("$i")
    done
else
    IFS=',' read -ra SEED_ARR <<< "$SEEDS"
fi

# Array to store commands
commands=()

# Create experiments for each combination
for src_dataset in "${SRC_DATASET_ARR[@]}"; do
    for trg_dataset in "${TRG_DATASET_ARR[@]}"; do
        # Skip iteration if src_dataset equals trg_dataset
        if [[ "$src_dataset" == "$trg_dataset" ]]; then
            continue
        fi
        for approach in "${APPROACH_ARR[@]}"; do
            for seed in "${SEED_ARR[@]}"; do
                if [[ -n "$LOG_KEYWORD" ]]; then
                    LOG_DIR="../results_sd_${src_dataset}_td_${trg_dataset}_${approach}_${LOG_KEYWORD}_6f_20p"
                else
                    LOG_DIR="../results_sd_${src_dataset}_td_${trg_dataset}_${approach}_6f_20p"
                fi
                mkdir -p "${LOG_DIR}"
                
                CMD="python main.py --src-dataset ${src_dataset} --trg-dataset ${trg_dataset} --approach ${approach} --seed ${seed} --log-dir ${LOG_DIR}${FLAT_FLAG} ${EXTRA_ARGS}"
                
                FULL_CMD="{ echo \"$(printf '=%.0s' {1..100})\" ; \
                    echo \"Command: ${CMD}\" ; \
                    echo \"$(printf '=%.0s' {1..100})\" >&2 ; \
                    echo \"Command: ${CMD}\" >&2 ; \
                    ${CMD} ; \
                } > \"${LOG_DIR}/output.log\" 2>> \"${LOG_DIR}/errors.log\""
                commands+=("$FULL_CMD")
            done
        done
    done
done

TOTAL=${#commands[@]}
echo "Total experiments: $TOTAL"

# Run experiments in parallel with progress reporting if GNU parallel is available.
if command -v parallel > /dev/null 2>&1; then
    printf "%s\n" "${commands[@]}" | parallel --progress --delay 10 --joblog ../log_dump/joblog.txt -j "$CPU_CORES"
else
    echo "GNU parallel not found; using xargs without progress tracking."
    printf "%s\n" "${commands[@]}" | xargs -I CMD -P "$CPU_CORES" bash -c 'sleep 10; CMD'
fi

echo "All experiments completed."